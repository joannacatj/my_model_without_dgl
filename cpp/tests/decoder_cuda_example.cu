// Decoder CUDA 验证：读取导出参数并运行 GraphTransformerDecoderCUDA。

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph_transformer_decoder_cuda.h"

namespace {
struct TensorMeta {
  std::vector<int64_t> shape;
  int64_t offset = 0;
  int64_t nbytes = 0;
};

std::string ReadText(const std::string& p) { std::ifstream f(p); if(!f) throw std::runtime_error("open "+p); std::stringstream s; s<<f.rdbuf(); return s.str(); }
std::vector<uint8_t> ReadBin(const std::string& p) { std::ifstream f(p,std::ios::binary); if(!f) throw std::runtime_error("open "+p); f.seekg(0,std::ios::end); size_t n=f.tellg(); f.seekg(0); std::vector<uint8_t>b(n); f.read((char*)b.data(),n); return b; }

std::vector<int64_t> ParseShape(const std::string& s){ std::vector<int64_t> o; std::regex re(R"((\d+))"); for(auto it=std::sregex_iterator(s.begin(),s.end(),re);it!=std::sregex_iterator();++it)o.push_back(std::stoll((*it)[1])); return o; }
TensorMeta Find(const std::string& m, const std::string& name){
  std::string esc=std::regex_replace(name,std::regex(R"([.^$|()\[\]{}*+?\\])"),R"(\$&)" );
  std::regex re("\\{[^\\{\\}]*\"name\"\\s*:\\s*\""+esc+"\"[^\\{\\}]*\"shape\"\\s*:\\s*\\[([^\\]]*)\\][^\\{\\}]*\"offset\"\\s*:\\s*(\\d+)[^\\{\\}]*\"nbytes\"\\s*:\\s*(\\d+)");
  std::smatch sm; if(!std::regex_search(m,sm,re)) throw std::runtime_error("tensor not found: "+name);
  return {ParseShape(sm[1]), std::stoll(sm[2]), std::stoll(sm[3])};
}
int64_t Cfg(const std::string& c,const std::string& k,int64_t d){ std::regex re("\""+k+"\"\\s*:\\s*(\\d+)"); std::smatch m; return std::regex_search(c,m,re)?std::stoll(m[1]):d; }

float* Load(const std::vector<uint8_t>& b, const TensorMeta& t){
  float* d=nullptr; cudaMalloc(&d,t.nbytes); cudaMemcpy(d,b.data()+t.offset,t.nbytes,cudaMemcpyHostToDevice); return d;
}
int64_t InferLayers(const std::string& m){ std::regex re(R"(decoder\.layers\.(\d+)\.attention\.wq\.weight)"); int64_t mx=-1; for(auto it=std::sregex_iterator(m.begin(),m.end(),re);it!=std::sregex_iterator();++it) mx=std::max(mx,std::stoll((*it)[1])); return mx+1; }

}  // namespace

int main(int argc, char** argv){
  using namespace neugn::decoder;
  std::string manifest="checkpoints/wikics/export_cpp/weights_manifest.json";
  std::string weights="checkpoints/wikics/export_cpp/graphdecoder_weights.bin";
  std::string config="checkpoints/wikics/export_cpp/export_config.json";
  for(int i=1;i+1<argc;i+=2){ std::string k=argv[i],v=argv[i+1]; if(k=="--manifest")manifest=v; if(k=="--weights")weights=v; if(k=="--config")config=v; }

  auto m = ReadText(manifest);
  auto b = ReadBin(weights);
  auto c = ReadText(config);

  DecoderWeights w;
  w.dim = Cfg(c,"dim",512);
  w.n_heads = Cfg(c,"n_heads",8);
  w.n_kv_heads = w.n_heads;
  w.max_spd = Cfg(c,"max_spd",20);
  w.sub_node_id_size = 64;
  w.pos_size = 1024;
  w.n_layers = InferLayers(m);

  auto in_w = Find(m,"decoder.input_projection.weight");
  auto in_b = Find(m,"decoder.input_projection.bias");
  w.input_projection = {.weight=Load(b,in_w), .bias=Load(b,in_b), .in_dim=in_w.shape[1], .out_dim=in_w.shape[0]};
  w.sos_token = Load(b, Find(m,"decoder.sos_token"));
  w.node_embeddings = Load(b, Find(m,"decoder.node_embeddings.ne"));
  w.pos_embeddings = Load(b, Find(m,"decoder.pos_embeddings.pe"));
  w.type_embeddings = Load(b, Find(m,"decoder.type_embeddings.weight"));
  w.spd_bias_embedding = Load(b, Find(m,"decoder.spd_bias_embedding.weight"));

  w.final_norm = {.weight=Load(b,Find(m,"decoder.norm.weight")), .eps=1e-5f};
  auto o1w=Find(m,"decoder.output.0.weight"), o1b=Find(m,"decoder.output.0.bias");
  auto o2w=Find(m,"decoder.output.2.weight"), o2b=Find(m,"decoder.output.2.bias");
  w.out1 = {.weight=Load(b,o1w), .bias=Load(b,o1b), .in_dim=o1w.shape[1], .out_dim=o1w.shape[0]};
  w.out2 = {.weight=Load(b,o2w), .bias=Load(b,o2b), .in_dim=o2w.shape[1], .out_dim=o2w.shape[0]};

  for(int64_t i=0;i<w.n_layers;++i){
    DecoderLayerParam L;
    auto q=Find(m,"decoder.layers."+std::to_string(i)+".attention.wq.weight");
    auto k=Find(m,"decoder.layers."+std::to_string(i)+".attention.wk.weight");
    auto v=Find(m,"decoder.layers."+std::to_string(i)+".attention.wv.weight");
    auto o=Find(m,"decoder.layers."+std::to_string(i)+".attention.wo.weight");
    L.wq={Load(b,q),nullptr,q.shape[1],q.shape[0]};
    L.wk={Load(b,k),nullptr,k.shape[1],k.shape[0]};
    L.wv={Load(b,v),nullptr,v.shape[1],v.shape[0]};
    L.wo={Load(b,o),nullptr,o.shape[1],o.shape[0]};

    auto w1=Find(m,"decoder.layers."+std::to_string(i)+".feed_forward.w1.weight");
    auto w2=Find(m,"decoder.layers."+std::to_string(i)+".feed_forward.w2.weight");
    auto w3=Find(m,"decoder.layers."+std::to_string(i)+".feed_forward.w3.weight");
    L.w1={Load(b,w1),nullptr,w1.shape[1],w1.shape[0]};
    L.w2={Load(b,w2),nullptr,w2.shape[1],w2.shape[0]};
    L.w3={Load(b,w3),nullptr,w3.shape[1],w3.shape[0]};

    L.attention_norm={Load(b,Find(m,"decoder.layers."+std::to_string(i)+".attention_norm.weight")),1e-5f};
    L.ffn_norm={Load(b,Find(m,"decoder.layers."+std::to_string(i)+".ffn_norm.weight")),1e-5f};
    w.layers.push_back(L);
  }

  int64_t B=1, Seq=3, Feat=w.input_projection.in_dim;
  std::vector<float> graph_feat(B*w.dim,0.01f), input_feat(B*Seq*Feat,0.02f);
  std::vector<int64_t> subnode{0,1,2}, tml{Seq}, spd{0,1,2,1,0,1,2,1,0};
  float *d_gf,*d_if,*d_out; int64_t *d_sub,*d_tml,*d_spd;
  cudaMalloc(&d_gf,sizeof(float)*graph_feat.size()); cudaMemcpy(d_gf,graph_feat.data(),sizeof(float)*graph_feat.size(),cudaMemcpyHostToDevice);
  cudaMalloc(&d_if,sizeof(float)*input_feat.size()); cudaMemcpy(d_if,input_feat.data(),sizeof(float)*input_feat.size(),cudaMemcpyHostToDevice);
  cudaMalloc(&d_sub,sizeof(int64_t)*subnode.size()); cudaMemcpy(d_sub,subnode.data(),sizeof(int64_t)*subnode.size(),cudaMemcpyHostToDevice);
  cudaMalloc(&d_tml,sizeof(int64_t)); cudaMemcpy(d_tml,tml.data(),sizeof(int64_t),cudaMemcpyHostToDevice);
  cudaMalloc(&d_spd,sizeof(int64_t)*spd.size()); cudaMemcpy(d_spd,spd.data(),sizeof(int64_t)*spd.size(),cudaMemcpyHostToDevice);
  cudaMalloc(&d_out,sizeof(float)*B*(Seq+1)*Feat);

  DecoderIO io{d_gf,d_if,d_sub,d_tml,d_spd,B,Seq,Feat,d_out};
  GraphTransformerDecoderCUDA dec(w);
  dec.Forward(io);
  cudaDeviceSynchronize();

  std::vector<float> out(B*(Seq+1)*Feat);
  cudaMemcpy(out.data(),d_out,sizeof(float)*out.size(),cudaMemcpyDeviceToHost);
  std::cout << "=== CUDA Trained NeuGN Decoder Output ===\n";
  std::cout << "decoder_output shape=("<<B<<","<<(Seq+1)<<","<<Feat<<")\n";
  std::cout << "decoder_output values=";
  for(float v: out) std::cout << " " << v;
  std::cout << "\n";
  return 0;
}
