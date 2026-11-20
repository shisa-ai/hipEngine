// Capture Muse Glimmer DFlash draft oracles from llama.cpp (read-only peer).
//
// Replicates the draft-dflash driver's prepare/draft sequence for one prompt:
// target prefill with layer-input extraction -> feature injection into the
// draft KV -> noise-block decode -> draft logits dump. Links the external
// llama.cpp HIP build; not part of the hipEngine runtime.
//
// Build:
//   hipcc -O2 -std=c++17 scripts/muse_glimmer_dflash_oracle.cpp \
//     -I~/llama.cpp/llama.cpp-hip/include -I~/llama.cpp/llama.cpp-hip/src \
//     -I~/llama.cpp/llama.cpp-hip/ggml/include \
//     -L~/llama.cpp/llama.cpp-hip/build/bin -lllama -lggml -lggml-base \
//     -Wl,-rpath,$HOME/llama.cpp/llama.cpp-hip/build/bin \
//     -o /tmp/muse_glimmer_dflash_oracle
//
// Run: /tmp/muse_glimmer_dflash_oracle <text.gguf> <dflash.gguf> <outdir>

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

static llama_batch make_batch(int32_t n_tokens, int32_t n_embd) {
    llama_batch b = llama_batch_init(n_tokens, n_embd, 1);
    return b;
}

static void batch_add_token(llama_batch & b, llama_token tok, llama_pos pos, bool logits) {
    int32_t i = b.n_tokens++;
    b.token[i] = tok;
    b.pos[i] = pos;
    b.n_seq_id[i] = 1;
    b.seq_id[i][0] = 0;
    b.logits[i] = logits ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <text.gguf> <dflash.gguf> <outdir>\n", argv[0]);
        return 2;
    }
    const char * tgt_path = argv[1];
    const char * dft_path = argv[2];
    const char * outdir   = argv[3];

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = getenv("GLIMMER_CPU") ? 0 : 99;
    llama_model * model_tgt = llama_model_load_from_file(tgt_path, mp);
    if (!model_tgt) { fprintf(stderr, "target load failed\n"); return 1; }
    llama_model * model_dft = llama_model_load_from_file(dft_path, mp);
    if (!model_dft) { fprintf(stderr, "draft load failed\n"); return 1; }

    const int n_embd = llama_model_n_embd(model_tgt);
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx_tgt = llama_init_from_model(model_tgt, cp);
    cp.ctx_other = ctx_tgt;
    llama_context * ctx_dft = llama_init_from_model(model_dft, cp);
    if (!ctx_tgt || !ctx_dft) { fprintf(stderr, "ctx init failed\n"); return 1; }
    llama_set_causal_attn(ctx_dft, false); // DFlash block attention is non-causal

    // extract layers from draft metadata
    char buf[256];
    std::vector<int> layers;
    if (llama_model_meta_val_str(model_dft, "dflash.target_layers", buf, sizeof(buf)) > 0) {
        // array metadata: read via kv api instead
    }
    // use the known Glimmer contract from the artifact scan; read count safely
    const int want[5] = {2, 14, 26, 38, 50};
    for (int l : want) { layers.push_back(l); llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) l, true); }
    const int n_taps = (int) layers.size();
    const int mask_token = 201818;
    const int block_size = 16;

    // prompt: BOS + "The capital of France is" (oracle fixture ids)
    std::vector<llama_token> ids = {200000, 954, 7963, 323, 11698, 373};
    const int n_ids = (int) ids.size();

    llama_batch bt = make_batch(n_ids, 0);
    for (int i = 0; i < n_ids; i++) batch_add_token(bt, ids[i], i, i == n_ids - 1);
    if (llama_decode(ctx_tgt, bt) != 0) { fprintf(stderr, "target decode failed\n"); return 1; }
    float * tgt_logits = llama_get_logits_ith(ctx_tgt, n_ids - 1);
    int tgt_argmax = (int) (std::max_element(tgt_logits, tgt_logits + n_vocab) - tgt_logits);

    // gather features [n_ids, n_taps*n_embd]
    std::vector<float> features((size_t) n_ids * n_taps * n_embd);
    for (int k = 0; k < n_taps; k++) {
        float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) layers[k]);
        if (!layer) { fprintf(stderr, "layer %d extraction failed\n", layers[k]); return 1; }
        for (int i = 0; i < n_ids; i++)
            memcpy(features.data() + (size_t) i * n_taps * n_embd + (size_t) k * n_embd,
                   layer + (size_t) i * n_embd, n_embd * sizeof(float));
    }

    // inject into draft KV
    llama_batch bi = make_batch(n_ids, n_taps * n_embd);
    memcpy(bi.embd, features.data(), features.size() * sizeof(float));
    bi.token = nullptr;
    for (int i = 0; i < n_ids; i++) {
        bi.pos[i] = i;
        bi.n_seq_id[i] = 1;
        bi.seq_id[i][0] = 0;
        bi.logits[i] = 0;
    }
    bi.n_tokens = n_ids;
    if (llama_decode(ctx_dft, bi) != 0) { fprintf(stderr, "inject decode failed\n"); return 1; }

    // noise block [anchor, MASK x 15] at positions n_ids..n_ids+15
    llama_batch bn = make_batch(block_size, 0);
    for (int i = 0; i < block_size; i++)
        batch_add_token(bn, i == 0 ? ids.back() : mask_token, n_ids + i, true);
    if (llama_decode(ctx_dft, bn) != 0) { fprintf(stderr, "noise decode failed\n"); return 1; }

    FILE * f = fopen((std::string(outdir) + "/draft_logits.f32").c_str(), "wb");
    std::vector<int> draft_ids;
    for (int i = 0; i < block_size; i++) {
        float * row = llama_get_logits_ith(ctx_dft, i);
        fwrite(row, sizeof(float), n_vocab, f);
        if (i >= 1) draft_ids.push_back((int) (std::max_element(row, row + n_vocab) - row));
    }
    fclose(f);

    FILE * fj = fopen((std::string(outdir) + "/draft_oracle.json").c_str(), "wb");
    fprintf(fj, "{\n  \"prompt_ids\": [200000, 954, 7963, 323, 11698, 373],\n");
    fprintf(fj, "  \"target_argmax\": %d,\n  \"block_size\": %d,\n  \"mask_token\": %d,\n", tgt_argmax, block_size, mask_token);
    fprintf(fj, "  \"draft_argmax\": [");
    for (size_t i = 0; i < draft_ids.size(); i++) fprintf(fj, "%s%d", i ? ", " : "", draft_ids[i]);
    fprintf(fj, "]\n}\n");
    fclose(fj);
    printf("target_argmax=%d draft=[", tgt_argmax);
    for (auto t : draft_ids) printf("%d ", t);
    printf("]\n");

    llama_batch_free(bt); llama_batch_free(bi); llama_batch_free(bn);
    llama_free(ctx_dft); llama_free(ctx_tgt);
    llama_model_free(model_dft); llama_model_free(model_tgt);
    llama_backend_free();
    return 0;
}