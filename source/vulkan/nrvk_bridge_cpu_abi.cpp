#include <windows.h>
#include <cstdio>
int main(int argc,char**argv){
  if(argc!=4){std::printf("usage: nrvk_bridge_cpu_abi <nrvk_bridge.dll> <port_network_bridge_backend.dll> <manifest_dir>\n");return 2;}
  HMODULE c=LoadLibraryA(argv[1]); if(!c){std::printf("FAIL load component %lu\n",GetLastError());return 1;}
  auto attach=(int(*)(void*,const void*,int(*)(const void*)))GetProcAddress(c,"NRVK_AttachToBridge");
  auto stop=(int(*)(void))GetProcAddress(c,"NRVK_Shutdown");
  if(!attach||!stop){std::printf("FAIL component exports\n");return 1;}
  if(stop()!=0){std::printf("FAIL shutdown idle\n");return 1;}
  if(attach(nullptr,nullptr,nullptr)!=-1){std::printf("FAIL attach null guard\n");return 1;}
  HMODULE b=LoadLibraryA(argv[2]); if(!b){std::printf("FAIL load backend %lu\n",GetLastError());return 1;}
  auto probe=(int(*)(const char*))GetProcAddress(b,"nrvk_backend_probe");
  auto last=(const char*(*)())GetProcAddress(b,"port_network_bridge_backend_last_error");
  if(!probe||!last){std::printf("FAIL backend exports\n");return 1;}
  int rc=probe(argv[3]);
  if(rc!=0){std::printf("FAIL backend probe rc=%d err=%s\n",rc,last());return 1;}
  FreeLibrary(b);FreeLibrary(c);
  std::printf("NRVK_CPU_ABI_PASS exports_ok idle_shutdown_ok null_attach_guard_ok model_plan_probe_ok\n");
  return 0;
}
