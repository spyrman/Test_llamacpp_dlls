#include <Windows.h>
#include <iostream>
#include <cstring>   // memset
#include <excpt.h>   // for __try/__except

// --- Minimal ggml/llama types (must match b5028 exactly!) ---

// opaque forward declarations
typedef void* ggml_backend_dev_t;
typedef void  llama_model;

// callback for llama progress
typedef bool(__cdecl* llama_progress_callback)(float progress, void* user_data);

// logging levels
enum ggml_log_level {
    GGML_LOG_LEVEL_ERROR = 2,
    GGML_LOG_LEVEL_WARN = 3,
    GGML_LOG_LEVEL_INFO = 4,
};

// this struct is taken verbatim from b5028's llama.h (in memory layout order)
struct llama_model_params {
    ggml_backend_dev_t* devices;                   // device list (NULL = auto)
    const void* tensor_buft_overrides;     // (not used here)
    int                   n_gpu_layers;              // how many layers to offload to GPU
    int                   split_mode;                // LLAMA_SPLIT_MODE_*
    int                   main_gpu;                  // which GPU to treat as “main”
    const float* tensor_split;              // custom splits if any
    llama_progress_callback progress_callback;       // progress cb (or NULL)
    void* progress_callback_user_data;
    const void* kv_overrides;              // overrides for KV tensors
    bool                  vocab_only;                // load only vocab?
    bool                  use_mmap;                  // mmap the model?
    bool                  use_mlock;                 // mlock the model?
    bool                  check_tensors;             // validate tensor shapes?
    bool                  padding1;                  // <— padding to align to 8 bytes
    int                   padding2;                  // <— more padding (total struct = 80 bytes)
};

// --- Function pointer typedefs ---
typedef void(__cdecl* t_llama_log_set)(void (*logger)(ggml_log_level, const char*, void*), void*);
typedef void(__cdecl* t_llama_backend_init)();
typedef llama_model_params(__cdecl* t_llama_model_default_params)();
typedef llama_model* (__cdecl* t_llama_model_load_from_file)(const char*, llama_model_params);
typedef void(__cdecl* t_ggml_backend_load_all)();

// a tiny logger to see internal llama messages
void llama_logger(ggml_log_level level, const char* msg, void*) {
    std::cout << "[llama] " << msg;
}

int main() {
    std::cout << "=== llama.cpp b5028 DLL Test ===\n";

    // 1) Load the two DLLs
    HMODULE hLlama = LoadLibraryA("llama.dll");
    HMODULE hGgml = LoadLibraryA("ggml.dll");
    if (!hLlama || !hGgml) {
        std::cerr << " Failed to load llama.dll or ggml.dll (GetLastError="
            << GetLastError() << ")\n";
        return 1;
    }
    std::cout << " DLLs loaded\n";

    // 2) Resolve all symbols
    auto llama_log_set = (t_llama_log_set)GetProcAddress(hLlama, "llama_log_set");
    auto llama_backend_init = (t_llama_backend_init)GetProcAddress(hLlama, "llama_backend_init");
    auto llama_model_default_params = (t_llama_model_default_params)
        GetProcAddress(hLlama, "llama_model_default_params");
    auto llama_model_load_from_file = (t_llama_model_load_from_file)
        GetProcAddress(hLlama, "llama_model_load_from_file");
    auto ggml_backend_load_all = (t_ggml_backend_load_all)GetProcAddress(hGgml, "ggml_backend_load_all");

    if (!llama_log_set || !llama_backend_init ||
        !llama_model_default_params || !llama_model_load_from_file ||
        !ggml_backend_load_all)
    {
        std::cerr << " One or more functions not found in the DLL exports\n";
        return 2;
    }
    std::cout << " Function bindings OK\n";

    // 3) Wire up our logger BEFORE anything else
    llama_log_set(llama_logger, nullptr);

    // 4) Load all available backends (CUDA, CPU, RPC...)
    std::cout << ">> ggml_backend_load_all()\n";
    ggml_backend_load_all();
    std::cout << " ggml backends loaded\n";

    // 5) Initialize llama internals
    std::cout << ">> llama_backend_init()\n";
    llama_backend_init();
    std::cout << " llama backend init\n";

    // 6) Fetch default model params, then tweak them
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 9999;
    params.use_mmap = true;    // mmap the file
    params.use_mlock = false;   // don’t lock in RAM
    params.n_gpu_layers = INT_MAX; // attempt to offload ALL layers to GPU
    params.check_tensors = false;
    params.vocab_only = false;

    // 7) Wrap the actual model load in SEH to catch access violations
    const char* path = "test_model.gguf";
    std::cout << ">> loading model: " << path << "\n";

    llama_model* model = nullptr;
    __try {
        model = llama_model_load_from_file(path, params);
        if (!model) {
            std::cerr << " llama_model_load_from_file returned NULL\n";
            return 3;
        }
    }
    __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION
        ? EXCEPTION_EXECUTE_HANDLER
        : EXCEPTION_CONTINUE_SEARCH)
    {
        std::cerr << " Access violation inside llama_model_load_from_file!\n";
        return 4;
    }

    std::cout << " Model loaded successfully!\n";
    
    std::cout << " Version b5028 \n";

    // (You can now call llama_model_free(model); if you want to unload it.)

    return 0;
}
