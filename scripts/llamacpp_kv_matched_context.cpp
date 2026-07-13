// Exact llama.cpp F16-vs-Q8_0 KV matched-context quality harness.
//
// This file intentionally uses only llama.cpp's public C API. It loads one
// model, runs an F16-KV reference, then runs a Q8_0-KV candidate while feeding
// the reference's greedy tokens. Only prompt-final plus decode-step logits are
// retained, avoiding llama-perplexity's context*vocab logit file.

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string model;
    std::string json_path;
    int32_t prompt_token_id = 9707;
    int32_t prompt_length = 131072;
    int32_t decode_steps = 16;
    int32_t ctx_size = 0;
    int32_t batch_size = 4096;
    int32_t ubatch_size = 512;
    int32_t n_gpu_layers = 99;
    int32_t threads = 16;
    std::string reference_cache = "f16";
    std::string candidate_cache = "q8_0";
    bool flash_attn = true;
    double kl_threshold = 0.05;
    double top1_threshold = 0.90;
};

struct RunResult {
    std::string cache_type;
    uint32_t actual_ctx_size = 0;
    int32_t n_vocab = 0;
    int32_t prompt_tokens = 0;
    int32_t decode_steps = 0;
    double context_create_seconds = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    bool finite_logits = true;
    std::vector<std::vector<float>> logits;
    std::vector<int32_t> top1_ids;
    std::vector<int32_t> decode_input_ids;
};

struct CompareResult {
    std::vector<double> kl;
    std::vector<bool> top1_matches;
    std::vector<int32_t> reference_top1;
    std::vector<int32_t> candidate_top1;
    std::vector<int32_t> candidate_reference_top1_rank;
    double mean_kl = 0.0;
    double max_kl = 0.0;
    double top1_agreement = 0.0;
    int32_t first_mismatch_index = -1;
};

[[noreturn]] void usage_error(const std::string & message) {
    throw std::runtime_error(message +
        "\nusage: llamacpp_kv_matched_context --model PATH --json PATH "
        "[--prompt-token-id 9707] [--prompt-length 131072] [--decode-steps 16] "
        "[--ctx-size N] [--batch-size 4096] [--ubatch-size 512] "
        "[--n-gpu-layers 99] [--threads 16] [--reference-cache f16] "
        "[--candidate-cache q8_0] [--flash-attn on|off]");
}

int32_t parse_i32(const std::string & text, const std::string & name) {
    size_t consumed = 0;
    long value = std::stol(text, &consumed, 10);
    if (consumed != text.size() || value < std::numeric_limits<int32_t>::min() ||
            value > std::numeric_limits<int32_t>::max()) {
        usage_error("invalid " + name + ": " + text);
    }
    return static_cast<int32_t>(value);
}

double parse_double(const std::string & text, const std::string & name) {
    size_t consumed = 0;
    double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        usage_error("invalid " + name + ": " + text);
    }
    return value;
}

bool parse_bool(const std::string & text, const std::string & name) {
    if (text == "on" || text == "true" || text == "1") {
        return true;
    }
    if (text == "off" || text == "false" || text == "0") {
        return false;
    }
    usage_error("invalid " + name + ": " + text);
}

Options parse_args(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage_error("help requested");
        }
        if (i + 1 >= argc) {
            usage_error("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--model") options.model = value;
        else if (arg == "--json") options.json_path = value;
        else if (arg == "--prompt-token-id") options.prompt_token_id = parse_i32(value, arg);
        else if (arg == "--prompt-length") options.prompt_length = parse_i32(value, arg);
        else if (arg == "--decode-steps") options.decode_steps = parse_i32(value, arg);
        else if (arg == "--ctx-size") options.ctx_size = parse_i32(value, arg);
        else if (arg == "--batch-size") options.batch_size = parse_i32(value, arg);
        else if (arg == "--ubatch-size") options.ubatch_size = parse_i32(value, arg);
        else if (arg == "--n-gpu-layers") options.n_gpu_layers = parse_i32(value, arg);
        else if (arg == "--threads") options.threads = parse_i32(value, arg);
        else if (arg == "--reference-cache") options.reference_cache = value;
        else if (arg == "--candidate-cache") options.candidate_cache = value;
        else if (arg == "--flash-attn") options.flash_attn = parse_bool(value, arg);
        else if (arg == "--kl-threshold") options.kl_threshold = parse_double(value, arg);
        else if (arg == "--top1-threshold") options.top1_threshold = parse_double(value, arg);
        else usage_error("unknown argument: " + arg);
    }
    if (options.model.empty()) usage_error("--model is required");
    if (options.json_path.empty()) usage_error("--json is required");
    if (options.prompt_token_id < 0) usage_error("--prompt-token-id must be non-negative");
    if (options.prompt_length <= 0) usage_error("--prompt-length must be positive");
    if (options.decode_steps < 0) usage_error("--decode-steps must be non-negative");
    if (options.batch_size <= 0 || options.ubatch_size <= 0 || options.threads <= 0) {
        usage_error("batch, ubatch, and thread counts must be positive");
    }
    if (options.ubatch_size > options.batch_size) usage_error("--ubatch-size cannot exceed --batch-size");
    if (options.ctx_size <= 0) options.ctx_size = options.prompt_length + options.decode_steps + 1;
    if (options.ctx_size < options.prompt_length + options.decode_steps) {
        usage_error("--ctx-size must cover prompt plus decode inputs");
    }
    return options;
}

enum ggml_type parse_cache_type(const std::string & value) {
    if (value == "f16") return GGML_TYPE_F16;
    if (value == "q8_0") return GGML_TYPE_Q8_0;
    throw std::runtime_error("unsupported KV cache type: " + value);
}

double seconds_since(const Clock::time_point & start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

int32_t argmax(const float * logits, int32_t n_vocab) {
    if (logits == nullptr || n_vocab <= 0) {
        throw std::runtime_error("invalid logits row");
    }
    return static_cast<int32_t>(std::max_element(logits, logits + n_vocab) - logits);
}

std::vector<float> copy_logits(llama_context * context, int32_t n_vocab, bool & finite) {
    const float * row = llama_get_logits_ith(context, -1);
    if (row == nullptr) {
        throw std::runtime_error("llama_get_logits_ith(-1) returned null");
    }
    std::vector<float> result(row, row + n_vocab);
    for (float value : result) {
        if (!std::isfinite(value)) {
            finite = false;
            break;
        }
    }
    return result;
}

RunResult run_context(
        llama_model * model,
        const Options & options,
        const std::string & cache_type,
        const std::vector<int32_t> * forced_inputs) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(options.ctx_size);
    params.n_batch = static_cast<uint32_t>(options.batch_size);
    params.n_ubatch = static_cast<uint32_t>(options.ubatch_size);
    params.n_seq_max = 1;
    params.n_threads = options.threads;
    params.n_threads_batch = options.threads;
    params.flash_attn_type = options.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.type_k = parse_cache_type(cache_type);
    params.type_v = parse_cache_type(cache_type);
    params.offload_kqv = true;
    params.no_perf = false;

    const auto create_start = Clock::now();
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model, params), llama_free);
    if (!context) {
        throw std::runtime_error("failed to create llama context for cache type " + cache_type);
    }

    RunResult result;
    result.cache_type = cache_type;
    result.actual_ctx_size = llama_n_ctx(context.get());
    result.prompt_tokens = options.prompt_length;
    result.decode_steps = options.decode_steps;
    result.context_create_seconds = seconds_since(create_start);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    result.n_vocab = llama_vocab_n_tokens(vocab);
    if (options.prompt_token_id >= result.n_vocab) {
        throw std::runtime_error("prompt token ID is outside model vocabulary");
    }
    if (forced_inputs != nullptr && static_cast<int32_t>(forced_inputs->size()) != options.decode_steps) {
        throw std::runtime_error("forced input count does not match decode steps");
    }

    std::vector<llama_token> prompt(static_cast<size_t>(options.prompt_length),
                                    static_cast<llama_token>(options.prompt_token_id));
    const auto prefill_start = Clock::now();
    int32_t offset = 0;
    while (offset < options.prompt_length) {
        const int32_t rows = std::min(options.batch_size, options.prompt_length - offset);
        llama_batch batch = llama_batch_get_one(prompt.data() + offset, rows);
        const int32_t rc = llama_decode(context.get(), batch);
        if (rc != 0) {
            std::ostringstream message;
            message << "llama_decode prefill failed at offset " << offset << " with code " << rc;
            throw std::runtime_error(message.str());
        }
        offset += rows;
    }
    result.prefill_seconds = seconds_since(prefill_start);
    result.logits.push_back(copy_logits(context.get(), result.n_vocab, result.finite_logits));
    result.top1_ids.push_back(argmax(result.logits.back().data(), result.n_vocab));

    const auto decode_start = Clock::now();
    for (int32_t step = 0; step < options.decode_steps; ++step) {
        const int32_t input_id = forced_inputs == nullptr
            ? result.top1_ids.back()
            : forced_inputs->at(static_cast<size_t>(step));
        result.decode_input_ids.push_back(input_id);
        llama_token token = static_cast<llama_token>(input_id);
        llama_batch batch = llama_batch_get_one(&token, 1);
        const int32_t rc = llama_decode(context.get(), batch);
        if (rc != 0) {
            std::ostringstream message;
            message << "llama_decode step " << step << " failed with code " << rc;
            throw std::runtime_error(message.str());
        }
        result.logits.push_back(copy_logits(context.get(), result.n_vocab, result.finite_logits));
        result.top1_ids.push_back(argmax(result.logits.back().data(), result.n_vocab));
    }
    result.decode_seconds = seconds_since(decode_start);
    return result;
}

CompareResult compare_logits(const RunResult & reference, const RunResult & candidate) {
    if (reference.n_vocab != candidate.n_vocab || reference.logits.size() != candidate.logits.size()) {
        throw std::runtime_error("reference/candidate logit shape mismatch");
    }
    CompareResult result;
    double kl_sum = 0.0;
    int64_t match_count = 0;
    for (size_t row_index = 0; row_index < reference.logits.size(); ++row_index) {
        const auto & p_logits = reference.logits[row_index];
        const auto & q_logits = candidate.logits[row_index];
        const double p_max = *std::max_element(p_logits.begin(), p_logits.end());
        const double q_max = *std::max_element(q_logits.begin(), q_logits.end());
        double p_sum = 0.0;
        double q_sum = 0.0;
        for (int32_t token = 0; token < reference.n_vocab; ++token) {
            p_sum += std::exp(static_cast<double>(p_logits[token]) - p_max);
            q_sum += std::exp(static_cast<double>(q_logits[token]) - q_max);
        }
        const double p_log_z = p_max + std::log(p_sum);
        const double q_log_z = q_max + std::log(q_sum);
        double kl = 0.0;
        for (int32_t token = 0; token < reference.n_vocab; ++token) {
            const double log_p = static_cast<double>(p_logits[token]) - p_log_z;
            const double log_q = static_cast<double>(q_logits[token]) - q_log_z;
            const double probability = std::exp(log_p);
            kl += probability * (log_p - log_q);
        }
        if (kl < 0.0 && kl > -1.0e-12) kl = 0.0;
        result.kl.push_back(kl);
        kl_sum += kl;
        result.max_kl = std::max(result.max_kl, kl);

        const int32_t ref_top1 = reference.top1_ids[row_index];
        const int32_t cand_top1 = candidate.top1_ids[row_index];
        result.reference_top1.push_back(ref_top1);
        result.candidate_top1.push_back(cand_top1);
        const bool matches = ref_top1 == cand_top1;
        result.top1_matches.push_back(matches);
        if (matches) {
            ++match_count;
        } else if (result.first_mismatch_index < 0) {
            result.first_mismatch_index = static_cast<int32_t>(row_index);
        }
        const float ref_token_logit = q_logits[ref_top1];
        int32_t rank = 1;
        for (float value : q_logits) {
            if (value > ref_token_logit) ++rank;
        }
        result.candidate_reference_top1_rank.push_back(rank);
    }
    if (!result.kl.empty()) {
        result.mean_kl = kl_sum / static_cast<double>(result.kl.size());
        result.top1_agreement = static_cast<double>(match_count) / static_cast<double>(result.kl.size());
    }
    return result;
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

template <typename T>
void write_number_array(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void write_bool_array(std::ostream & out, const std::vector<bool> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << (values[i] ? "true" : "false");
    }
    out << ']';
}

void write_run_json(std::ostream & out, const RunResult & run) {
    out << '{'
        << "\"cache_type\":\"" << json_escape(run.cache_type) << "\","
        << "\"actual_ctx_size\":" << run.actual_ctx_size << ','
        << "\"n_vocab\":" << run.n_vocab << ','
        << "\"finite_logits\":" << (run.finite_logits ? "true" : "false") << ','
        << "\"context_create_seconds\":" << run.context_create_seconds << ','
        << "\"prefill_seconds\":" << run.prefill_seconds << ','
        << "\"prefill_tok_s\":"
        << (run.prefill_seconds > 0.0 ? static_cast<double>(run.prompt_tokens) / run.prefill_seconds : 0.0) << ','
        << "\"decode_seconds\":" << run.decode_seconds << ','
        << "\"decode_tok_s\":"
        << (run.decode_seconds > 0.0 ? static_cast<double>(run.decode_steps) / run.decode_seconds : 0.0) << ','
        << "\"top1_ids\":";
    write_number_array(out, run.top1_ids);
    out << ",\"decode_input_ids\":";
    write_number_array(out, run.decode_input_ids);
    out << '}';
}

void write_json(
        const Options & options,
        const RunResult & reference,
        const RunResult & candidate,
        const CompareResult & comparison,
        double model_load_seconds) {
    std::ofstream out(options.json_path);
    if (!out) throw std::runtime_error("failed to open JSON output: " + options.json_path);
    out << std::setprecision(17);
    const bool passed = reference.finite_logits && candidate.finite_logits &&
        comparison.mean_kl <= options.kl_threshold && comparison.top1_agreement >= options.top1_threshold;
    out << '{'
        << "\"schema\":1,"
        << "\"status\":\"" << (passed ? "accepted" : "rejected_correctness") << "\","
        << "\"mode\":\"llamacpp_kv_matched_context\","
        << "\"performance_claim\":false,"
        << "\"model\":\"" << json_escape(options.model) << "\","
        << "\"prompt_token_id\":" << options.prompt_token_id << ','
        << "\"prompt_length\":" << options.prompt_length << ','
        << "\"decode_steps\":" << options.decode_steps << ','
        << "\"positions\":" << comparison.kl.size() << ','
        << "\"batch_size\":" << options.batch_size << ','
        << "\"ubatch_size\":" << options.ubatch_size << ','
        << "\"n_gpu_layers\":" << options.n_gpu_layers << ','
        << "\"threads\":" << options.threads << ','
        << "\"flash_attn\":" << (options.flash_attn ? "true" : "false") << ','
        << "\"model_load_seconds\":" << model_load_seconds << ','
        << "\"quality_thresholds\":{\"kl_mean_max\":" << options.kl_threshold
        << ",\"top1_agreement_min\":" << options.top1_threshold << "},"
        << "\"reference\":";
    write_run_json(out, reference);
    out << ",\"candidate\":";
    write_run_json(out, candidate);
    out << ",\"matched_context\":{"
        << "\"semantics\":\"candidate consumes F16-reference seed and generated tokens\","
        << "\"all_logit_positions_share_token_inputs\":true,"
        << "\"mean_kl\":" << comparison.mean_kl << ','
        << "\"max_kl\":" << comparison.max_kl << ','
        << "\"top1_agreement\":" << comparison.top1_agreement << ','
        << "\"passed\":" << (passed ? "true" : "false") << ','
        << "\"first_top1_mismatch\":";
    if (comparison.first_mismatch_index < 0) {
        out << "null";
    } else {
        const int32_t index = comparison.first_mismatch_index;
        out << "{\"index\":" << index
            << ",\"reference\":" << comparison.reference_top1[static_cast<size_t>(index)]
            << ",\"candidate\":" << comparison.candidate_top1[static_cast<size_t>(index)] << '}';
    }
    out << ",\"kl\":";
    write_number_array(out, comparison.kl);
    out << ",\"top1_matches\":";
    write_bool_array(out, comparison.top1_matches);
    out << ",\"reference_top1\":";
    write_number_array(out, comparison.reference_top1);
    out << ",\"candidate_top1\":";
    write_number_array(out, comparison.candidate_top1);
    out << ",\"candidate_reference_top1_rank\":";
    write_number_array(out, comparison.candidate_reference_top1_rank);
    out << "},\"notes\":["
        << "\"F16 and Q8_0 designate llama.cpp KV cache types; model weights are identical.\","
        << "\"Only prompt-final and decode-step full logits are retained in host memory.\","
        << "\"This is a correctness diagnostic, not a throughput claim.\"]}"
        << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_args(argc, argv);
        ggml_backend_load_all();
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.n_gpu_layers;
        model_params.main_gpu = 0;
        const auto load_start = Clock::now();
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(options.model.c_str(), model_params), llama_model_free);
        if (!model) throw std::runtime_error("failed to load model");
        const double model_load_seconds = seconds_since(load_start);

        std::cerr << "running F16 reference..." << std::endl;
        RunResult reference = run_context(model.get(), options, options.reference_cache, nullptr);
        std::vector<int32_t> forced_inputs;
        forced_inputs.reserve(static_cast<size_t>(options.decode_steps));
        for (int32_t step = 0; step < options.decode_steps; ++step) {
            forced_inputs.push_back(reference.top1_ids.at(static_cast<size_t>(step)));
        }

        std::cerr << "running Q8_0 candidate with reference-token forcing..." << std::endl;
        RunResult candidate = run_context(model.get(), options, options.candidate_cache, &forced_inputs);
        CompareResult comparison = compare_logits(reference, candidate);
        write_json(options, reference, candidate, comparison, model_load_seconds);

        model.reset();
        llama_backend_free();
        std::cerr << "mean KL=" << comparison.mean_kl
                  << " max KL=" << comparison.max_kl
                  << " top1=" << comparison.top1_agreement << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
