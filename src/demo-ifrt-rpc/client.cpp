#include <iostream>
#include <string>
#include <vector>
#include "xla/tsl/platform/statusor.h"
// #include "xla/python/ifrt_proxy/client/client.h"
#include "xla/python/ifrt_proxy/client/registry.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"


int main(int argc, const char** argv) {
    std::vector<std::string> args;
    args.assign(argv + 1, argv + argc);

    auto address = args[0];
    std::cout << "address: " << address << std::endl;

    // address must be in standard URI format
    // it is passed to `::grpc::ServerBuilder::AddListentingPort`
    // https://grpc.github.io/grpc/cpp/classgrpc_1_1_server_builder.html#ad8c7eff4f5747333b7ffbb09cef71b5e
    // auto grpc_server = MyValueOrThrow(
    //     xla::ifrt::proxy::CreateFromIfrtClientFactory(
    //         address,
    //         []() {
    //             xla::CpuClientOptions options;
    //             options.asynchronous = true;
    //             options.cpu_device_count = 1;

    //             TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> pjrt_cpu_client, xla::GetXlaPjrtCpuClient(options));
    //             return xla::ifrt::PjRtClient::Create(std::move(pjrt_cpu_client));
    //         }
    //     )
    // ).release();
    auto client = xla::ifrt::proxy::CreateClient("grpc://" + address);
    std::cout << "success initializing! now wait work...";

    std::cout << "Goodbye!" << std::endl;
}
