#include <iostream>
#include <string>
#include <vector>
#include "xla/tsl/platform/statusor.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"

extern "C" void (*ReactantThrowError)(const char *) = nullptr;

template <typename T>
T MyValueOrThrow(absl::StatusOr<T> v)
{
    if (ReactantThrowError)
    {
        if (!v.ok())
        {
            ReactantThrowError(v.status().ToString().c_str());
            throw xla::XlaRuntimeError(v.status().ToString().c_str());
        }
        return std::move(v).value();
    }
    else
    {
        return xla::ValueOrThrow(std::move(v));
    }
}

int main(int argc, const char** argv) {
    std::vector<std::string> args;
    args.assign(argv + 1, argv + argc);

    auto address = args[0];
    std::cout << "address: " << address << std::endl;

    // address must be in standard URI format
    // it is passed to `::grpc::ServerBuilder::AddListentingPort`
    // https://grpc.github.io/grpc/cpp/classgrpc_1_1_server_builder.html#ad8c7eff4f5747333b7ffbb09cef71b5e
    auto grpc_server = MyValueOrThrow(
        xla::ifrt::proxy::GrpcServer::CreateFromIfrtClientFactory(
            address,
            []() -> absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> {
                xla::CpuClientOptions options;
                options.asynchronous = true;
                options.cpu_device_count = 1;

                TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> pjrt_cpu_client, xla::GetXlaPjrtCpuClient(options));
                return std::shared_ptr<xla::ifrt::Client>(
                    xla::ifrt::PjRtClient::Create(std::move(pjrt_cpu_client)).release()
                );
            }
        )
    ); // .release();

    std::cout << "success initializing! now wait work...";

    grpc_server->Wait();

    std::cout << "Goodbye!" << std::endl;
}
