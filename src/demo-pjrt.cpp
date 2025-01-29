#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/CAPI/Support.h"

#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/LLVMIRToNVVMTranslation.h"

// #include "Enzyme/MLIR/Dialect/Dialect.h"
// #include "Enzyme/MLIR/Dialect/Ops.h"
// #include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
// #include "Enzyme/MLIR/Passes/Passes.h"

#include "xla/pjrt/status_casters.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"

using namespace xla;
using namespace mlir;

// This is set by Reactant.jl on startup to allow throwing errors back to Julia.
extern "C" extern void (*ReactantThrowError)(const char *) = nullptr;

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

void prepareRegistry(mlir::DialectRegistry &registry);

// Initializes the MLIR registry and passes.
extern "C" void InitializeRegistryAndPasses(MlirDialectRegistry creg)
{
    mlir::DialectRegistry &registry = *unwrap(creg);
    prepareRegistry(registry);

    // mlir::registerenzymePasses();
    // registerenzymexlaPasses();

    // Register the standard passes we want.
    mlir::registerCSEPass();
    mlir::registerConvertAffineToStandardPass();
    mlir::registerSCCPPass();
    mlir::registerInlinerPass();
    mlir::registerCanonicalizerPass();
    mlir::registerSymbolDCEPass();
    mlir::registerLoopInvariantCodeMotionPass();
    // mlir::registerConvertSCFToOpenMPPass();
    // mlir::affine::registerAffinePasses();
    mlir::registerReconcileUnrealizedCasts();

    mlir::registerLLVMDialectImport(registry);
    // mlir::registerNVVMDialectImport(registry);
    mlir::LLVM::registerInlinerInterface(registry);

    // Transform dialect and extensions.
    mlir::transform::registerInterpreterPass();
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
    prepareRegistry(registry);
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
extern "C" PjRtBuffer *ArrayFromHostBuffer(PjRtClient *client, void *data, uint64_t ptype, size_t dim, int64_t *cshape, PjRtDevice *device)
{
    auto primtype = (xla::PrimitiveType)ptype;
    absl::Span<const int64_t> shape(cshape, dim);
    PjRtClient::HostBufferSemantics semantics =
        PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;
    // xla::Layout layout(col_major(dim));
    // auto buffer = xla::MyValueOrThrow(client->BufferFromHostBuffer(data,
    // primtype, shape, /*byte_strides*/{},  semantics, /*ondone*/{}, device,
    // &layout));
    auto buffer = MyValueOrThrow(
        client->BufferFromHostBuffer(data, primtype, shape, /*byte_strides*/ {},
                                     semantics, /*ondone*/ {}, device));
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
    MlirDialectRegistry registry = MlirDialectRegistryCreate();
    InitializeRegistryAndPasses(registry);

    // 2. init PjRt client (CPU)
    uint8_t async = false;
    int node_id = 0;
    int num_nodes = 1;
    xla::PjRtClient *client = MakeCPUClient(async, node_id, num_nodes);

    // 3. parse MLIR
    MlirContext mlir_ctx = mlirContextCreateWithRegistry(registry, false);
    RegisterDialects(mlir_ctx);
    const char *mlir_code = "
        module
    {
        func.func @main(% arg0 : tensor<4x4xf64>)->tensor<4x4xf64>
        {
            % 0 = stablehlo.sine % arg0 : tensor<4x4xf64> return % 0 : tensor<4x4xf64>
        }
    }
    ";
        MlirModule mlir_mod = mlirModuleCreateParse(mlir_code, registry);

    // 4. compile MLIR module to XLA executable
    xla::PjRtLoadedExecutable *loaded_exec = ClientCompile(client, mlir_code);

    // 5. create input array
    float64_t *ptr = new float64_t[16];
    int64_t shape[2] = {4, 4};
    size_t dim = 2;
    uint64_t prim_type = 12; // float64
    for (int i = 0; i < 16; i++)
    {
        ptr[i] = 1.0 + i;
    }

    int default_device_idx = 0;
    xla::PjRtDevice *device = ClientGetDevice(client, default_device_idx);

    xla::PjRtBuffer *buffer = ArrayFromHostBuffer(client, ptr, prim_type, dim, &shape, device);

    // 6. execute computation
    int num_args = 1;
    PjRtBuffer **op_args = new PjRtBuffer *[num_args];
    op_args[0] = buffer;
    uint8_t *is_arg_donatable = new uint8_t[num_args];
    is_arg_donatable[0] = false;
    int num_results = 1;
    PjRtBuffer **op_results = new PjRtBuffer *[num_results];
    uint8_t futures;
    PjRtFuture **future_results = new PjRtFuture *[num_results];
    XLAExecute(loaded_exec, num_args, op_args, is_arg_donatable, num_results, op_results, &futures, future_results);

    // 7. print results
    float64_t *ptr_result = new float64_t[16];
    BufferToHost(op_results[0], ptr_result);

    for (int i = 0; i < 16; i++)
    {
        printf("[%d] sin(%f) = %f\n", i, ptr[i], ptr_result[i]);
    }

    // 8. free memory
    FreeClient(client);
    delete[] ptr;
    delete[] ptr_result;
}
