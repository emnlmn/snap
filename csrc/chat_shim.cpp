// C ABI shim over llama.cpp's common/chat.h — the same jinja renderer
// llama-server uses, including enable_thinking support. libllama-common.a is
// already linked by llama-cpp-sys-2, so symbols resolve at final link.

#include <cstdlib>
#include <cstring>
#include <string>

#include "chat.h"

extern "C" void * snap_chat_templates_init(
    struct llama_model * model,
    const char * tmpl_override,
    const char * bos_override,
    const char * eos_override) {
    try {
        auto p = common_chat_templates_init(
            model,
            tmpl_override ? tmpl_override : "",
            bos_override ? bos_override : "",
            eos_override ? eos_override : "");
        return static_cast<void *>(p.release());
    } catch (...) {
        return nullptr;
    }
}

extern "C" char * snap_chat_apply(
    void * tmpls_ptr,
    const char * system_content,
    const char * user_content) {
    auto * tmpls = static_cast<common_chat_templates *>(tmpls_ptr);
    if (!tmpls || !system_content || !user_content) {
        return nullptr;
    }
    try {
        common_chat_msg sys_msg;
        sys_msg.role = "system";
        sys_msg.content = system_content;
        common_chat_msg user_msg;
        user_msg.role = "user";
        user_msg.content = user_content;

        common_chat_templates_inputs inputs;
        inputs.use_jinja = true;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking = false; // snap never enables thinking
        inputs.messages = {sys_msg, user_msg};

        const auto params = common_chat_templates_apply(tmpls, inputs);
        char * out = static_cast<char *>(std::malloc(params.prompt.size() + 1));
        if (!out) {
            return nullptr;
        }
        std::memcpy(out, params.prompt.c_str(), params.prompt.size() + 1);
        return out;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void snap_chat_templates_free(void * tmpls_ptr) {
    if (tmpls_ptr) {
        common_chat_templates_free(static_cast<common_chat_templates *>(tmpls_ptr));
    }
}

extern "C" void snap_str_free(char * p) {
    std::free(p);
}
