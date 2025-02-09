#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/CAPI/Support.h"

#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
// #include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
// #include "mlir/Target/LLVMIR/Dialect/NVVM/LLVMIRToNVVMTranslation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

// #include "Enzyme/MLIR/Dialect/Dialect.h"
// #include "Enzyme/MLIR/Dialect/Ops.h"
// #include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
// #include "Enzyme/MLIR/Passes/Passes.h"

#include "xla/tsl/concurrency/ref_count.h"

#include "xla/pjrt/status_casters.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"

#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/ifrt/host_callback.h"

#include "xla/python/ifrt/hlo/hlo_program.h"

#include <chrono>
#include <iostream>
#include <thread>


using namespace xla;
using namespace mlir;

// Julia is row-major (like Fortran) and 1-indexed.
// https://openxla.org/xla/shapes
// This minor-to-major dimension order of 0 up to N-1 is akin to column-major
// (at rank 2). Assuming a monotonic ordering of dimensions, another way we may
// refer to this layout in the code is simply "dim 0 is minor".
std::vector<int64_t> col_major(int64_t dim)
{
    std::vector<int64_t> minor_to_major;
    for (int i = 0; i < dim; i++)
    {
        minor_to_major.push_back(i); // dim-1-i);
                                     // minor_to_major.push_back(dim-1-i);
    }
    return minor_to_major;
}

std::vector<int64_t> row_major(int64_t dim)
{
    std::vector<int64_t> minor_to_major;
    for (int i = 0; i < dim; i++)
    {
        minor_to_major.push_back(dim - 1 - i);
    }
    return minor_to_major;
}

// This is set by Reactant.jl on startup to allow throwing errors back to Julia.
extern "C" void (*ReactantThrowError)(const char *) = nullptr;

// Utilities for `StatusOr`.
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

// void prepareRegistry(mlir::DialectRegistry &registry);

// Initializes the MLIR registry and passes.
extern "C" void InitializeRegistryAndPasses(MlirDialectRegistry creg)
{
    mlir::DialectRegistry &registry = *unwrap(creg);
    // prepareRegistry(registry);

    // mlir::registerenzymePasses();
    // registerenzymexlaPasses();

    // Register the standard passes we want.
    // mlir::registerCSEPass();
    // mlir::registerConvertAffineToStandardPass();
    // mlir::registerSCCPPass();
    // mlir::registerInlinerPass();
    // mlir::registerCanonicalizerPass();
    // mlir::registerSymbolDCEPass();
    // mlir::registerLoopInvariantCodeMotionPass();
    // mlir::registerConvertSCFToOpenMPPass();
    // mlir::affine::registerAffinePasses();
    // mlir::registerReconcileUnrealizedCasts();

    // mlir::registerLLVMDialectImport(registry);
    // mlir::registerNVVMDialectImport(registry);
    mlir::LLVM::registerInlinerInterface(registry);

    // Transform dialect and extensions.
    // mlir::transform::registerInterpreterPass();
    // mlir::enzyme::registerGenerateApplyPatternsPass();
    // mlir::enzyme::registerRemoveTransformPass();
}

// Creates a CPU PjRt client.
extern "C" PjRtClient *MakeCPUClient(uint8_t asynchronous, int node_id, int num_nodes)
{
    CpuClientOptions options;
    // options.kv_store = "etcd";
    options.process_id = node_id;
    // options.num_nodes = num_nodes;
    // options.collectives = num_nodes;
    options.asynchronous = asynchronous != 0;
    auto client = MyValueOrThrow(GetTfrtCpuClient(options));
    return client.release();
}



// Registers the MLIR dialects.
extern "C" void RegisterDialects(MlirContext cctx)
{
    mlir::MLIRContext &context = *unwrap(cctx);
    DialectRegistry registry;
    // prepareRegistry(registry);
    context.appendDialectRegistry(registry);
    context.loadDialect<mlir::arith::ArithDialect>();
    // context.loadDialect<mlir::enzyme::EnzymeDialect>();
    // context.loadDialect<mlir::enzymexla::EnzymeXLADialect>();
    // context.loadDialect<mlir::triton::TritonDialect>();
    // context.loadDialect<mlir::tpu::TPUDialect>();
    context.loadDialect<mlir::tensor::TensorDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::mhlo::MhloDialect>();
    context.loadDialect<mlir::stablehlo::StablehloDialect>();
    context.loadDialect<mlir::chlo::ChloDialect>();
}

// Compiles an MLIR module to an XLA executable (i.e. PjRtClient::Compile)
extern "C" xla::PjRtLoadedExecutable *ClientCompile(PjRtClient *client, MlirModule cmod)
{
    auto program =
        std::make_unique<xla::ifrt::HloProgram>(cast<ModuleOp>(*unwrap(cmod)));

    CompileOptions options;
    // options.argument_layouts;
    // options.executable_build_options.set_device_ordinal();
    // options.executable_build_options.set_result_layout();

    auto addressable_devices = client->addressable_devices();
    if (!addressable_devices.empty())
    {
        int device_ordinal = options.executable_build_options.device_ordinal();
        if (device_ordinal < 0)
        {
            device_ordinal = 0;
        }
        assert(device_ordinal < addressable_devices.size());
        auto stats = addressable_devices[device_ordinal]->GetAllocatorStats();
        if (stats.ok() && stats->bytes_limit)
        {
            options.executable_build_options.set_device_memory_size(
                *stats->bytes_limit);
        }
    }
    auto exec =
        MyValueOrThrow(client->Compile(cast<ModuleOp>(*unwrap(cmod)), options));
    return exec.release();
}

// Gets a device from a client (i.e. PjRtClient::LookupDevice)
extern "C" PjRtDevice *ClientGetDevice(PjRtClient *client, int device_id)
{
    return MyValueOrThrow(client->LookupDevice(PjRtGlobalDeviceId(device_id)));
}

// Creates an XLA buffer from a host buffer (i.e. PjRtClient::BufferFromHostBuffer)
extern "C" PjRtBuffer *ArrayFromHostBuffer(PjRtClient *client, void *data,
                                           uint64_t ptype, size_t dim,
                                           int64_t *cshape,
                                           PjRtDevice *device) {
  auto primtype = (xla::PrimitiveType)ptype;
  absl::Span<const int64_t> shape(cshape, dim);
  PjRtClient::HostBufferSemantics semantics =
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
  // xla::Layout layout(col_major(dim));
  // auto buffer = xla::MyValueOrThrow(client->BufferFromHostBuffer(data,
  // primtype, shape, /*byte_strides*/{},  semantics, /*ondone*/{}, device,
  // &layout));
  const xla::Layout* layout = nullptr;
  auto buffer = MyValueOrThrow(
      client->BufferFromHostBuffer(data, primtype, shape, /*byte_strides*/ {},
                                   semantics, /*ondone*/ {}, *device->default_memory_space(), layout));
  auto bres = buffer.release();
  return bres;
}

// Executes an XLA executable (i.e. PjRtLoadedExecutable::Execute)
extern "C" void XLAExecute(xla::PjRtLoadedExecutable *exec, int num_args, PjRtBuffer **op_args, uint8_t *is_arg_donatable, int num_results, PjRtBuffer **op_results, uint8_t *futures, PjRtFuture<> **future_results)
{
    std::vector<std::vector<PjRtBuffer *>> argument_handles;
    argument_handles.emplace_back(op_args, op_args + num_args);

    ExecuteOptions options;

    for (size_t i = 0; i < num_args; i++)
    {
        if (!is_arg_donatable[i])
            options.non_donatable_input_indices.insert((int)i);
    }
    options.untuple_result = true;
    std::optional<std::vector<PjRtFuture<>>> returned_futures;
    auto results = MyValueOrThrow(
        exec->Execute(static_cast<absl::Span<const std::vector<PjRtBuffer *>>>(
                          argument_handles),
                      options, returned_futures));

    assert(results.size() == 1);

    if (results[0].size() != num_results)
    {
        llvm::errs() << " results.size()=" << results.size()
                     << " num_results=" << num_results << "\n";
    }
    assert(results[0].size() == num_results);
    if (returned_futures)
    {
        *futures = true;
        assert(returned_futures->size() == num_results);
        for (size_t i = 0; i < num_results; i++)
        {
            future_results[i] = new PjRtFuture<>((*returned_futures)[i]);
        }
    }
    else
    {
        *futures = false;
    }

    for (size_t i = 0; i < num_results; i++)
    {
        op_results[i] = results[0][i].release();
    }
}

extern "C" void BufferToHost(PjRtBuffer *buffer, void *data)
{
    Shape shape(MyValueOrThrow(buffer->HostShape()));
    /// Grumpily the cpu copy code does not respect layout and does a raw copy
    /// For now, we assume a non-julia row major ordering
    /// If in the future it supports col_major we can swap to that.
    *shape.mutable_layout() = xla::Layout(row_major(shape.dimensions_size()));
    MutableBorrowingLiteral literal((const char *)data, shape);
    auto status = buffer->ToLiteralSync(&literal);
    if (!status.ok())
    {
        printf("error copying to host: %s\n", status.ToString().c_str());
    }
}

extern "C" void FreeClient(PjRtClient *client) { delete client; }

int main()
{
    // 1. init MLIR registry and passes
    MlirDialectRegistry registry = mlirDialectRegistryCreate();
    InitializeRegistryAndPasses(registry);

    // 2. init PjRt client (CPU) & IFRT client from it (PjRt backend)
    uint8_t async = false;
    int node_id = 0;
    int num_nodes = 1;
    auto pjrt_client = std::shared_ptr<xla::PjRtClient>(MakeCPUClient(async, node_id, num_nodes));
    
    xla::ifrt::PjRtClient::CreateOptions options = {pjrt_client};
    xla::ifrt::PjRtClient* ifrt_client = MyValueOrThrow(xla::ifrt::PjRtClient::Create(options)).release();

    // 3. parse MLIR
    MlirContext mlir_ctx = mlirContextCreateWithRegistry(registry, false);
    RegisterDialects(mlir_ctx);
    const char *mlir_code_cstr =
        "module {\n"
        "func.func @main(\%arg0 : tensor<4x4xf64>)->tensor<4x4xf64> {\n"
        "    \%0 = stablehlo.sine \%arg0 : tensor<4x4xf64> return \%0 : tensor<4x4xf64>\n"
        "}\n"
        "}\n\0";
    MlirStringRef mlir_code = mlirStringRefCreateFromCString(mlir_code_cstr);
    MlirModule mlir_mod = mlirModuleCreateParse(mlir_ctx, mlir_code);
    mlir::ModuleOp mlir_mod_op = cast<ModuleOp>(*unwrap(mlir_mod));

    // 4. compile MLIR module to XLA executable
    xla::CompileOptions compile_options;
    xla::ifrt::LoadedExecutable *loaded_exec = MyValueOrThrow(xla::ifrt::PjRtLoadedExecutable::Create(ifrt_client, mlir_mod_op, compile_options, std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>())).release();

    // 5. create input array (use single-shard for now)
    double *ptr = new double[16];
    int64_t shape[2] = {4, 4};
    size_t dim = 2;
    uint64_t prim_type = 12; // float64
    for (int i = 0; i < 16; i++)
    {
        ptr[i] = 1.0 + i;
    }

    int default_device_idx = 0;
    xla::PjRtDevice *device = ClientGetDevice(pjrt_client.get(), default_device_idx);
    auto buffer = std::shared_ptr<xla::PjRtBuffer>(ArrayFromHostBuffer(pjrt_client.get(), ptr, prim_type, dim, shape, device));
    tsl::RCReference<xla::ifrt::PjRtArray> ifrt_input_array = MyValueOrThrow(xla::ifrt::PjRtArray::Create(ifrt_client, buffer));

    // 6. execute computation
    std::vector<tsl::RCReference<xla::ifrt::Array>> args;
    args.emplace_back(ifrt_input_array);

    xla::ifrt::ExecuteOptions exec_options;
    xla::ifrt::LoadedExecutable::ExecuteResult result = MyValueOrThrow(loaded_exec->Execute(static_cast<absl::Span<tsl::RCReference<xla::ifrt::Array>>>(args), exec_options, /* devices */ std::nullopt));

    // sync: block until done
    // using namespace std::chrono_literals;
    // std::this_thread::sleep_for(2000ms);

    // 7. print results
    double *ptr_result = new double[16];
    // BufferToHost(result.outputs[0]->pjrt_buffers()[0].get(), ptr_result);
    result.outputs[0]->CopyToHostBuffer(ptr_result, std::nullopt, xla::ifrt::ArrayCopySemantics::kAlwaysCopy);

    for (int i = 0; i < 16; i++)
    {
        printf("[%d] sin(%f) = %f\n", i, ptr[i], ptr_result[i]);
    }

    // 8. free memory
    delete loaded_exec;
    delete ifrt_client;
    // FreeClient(client);
    // delete[] ptr;
    // delete[] ptr_result;
}
