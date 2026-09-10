# PF-0 natural-source provenance

Prompt-body excerpts for `benchmarks/fixtures/qwen4exp_natural_ar_pf0.json`
(built by `scripts/build_qwen4exp_natural_ar_fixture.py`). Each `<category>-<n>.txt`
file is the exact chat-prompt body for the fixture case of the same id; the
Qwen chat template is added by the builder, not stored here.

All material is natural, unmodified prose/source from public or liberally
licensed origins. No text was generated, padded, or repeated; spans are
contiguous excerpts, cut at paragraph boundaries (code at line boundaries).

| Case files | Work | Origin | License |
| --- | --- | --- | --- |
| `code-1..3.txt` | CPython `Lib/dataclasses.py` (default branch, fetched 2026-09-03) | https://github.com/python/cpython | PSF License Version 2 |
| `general_en-1..3.txt` | *Walden, and On The Duty Of Civil Disobedience*, Henry David Thoreau | Project Gutenberg eBook #205, https://www.gutenberg.org/ebooks/205 | Public domain |
| `general_ja-1..3.txt` | 『こころ』 (*Kokoro*), 夏目 漱石 (Natsume Sōseki, 1914) | 青空文庫 (Aozora Bunko) card 773, `773_ruby_5968.zip`, https://www.aozora.gr.jp/cards/000148/card773.html | Public domain (author died 1916) |
| `mixed_ja_en-1..3.txt` | ruby/ruby `README.ja.md` | https://github.com/ruby/ruby | Ruby License / 2-clause BSDL (dual) |

Normalization applied at excerpt time (declaration, not modification of
content):

- *Kokoro*: Shift-JIS → UTF-8; Aozora ruby annotations (`《…》`, `｜`),
  input-notes (`［＃…］`), the symbol-key header, and the publication footer
  removed. Remaining text is the novel as printed.
- *Walden*: Project Gutenberg header/footer and table of contents removed;
  excerpt starts at the chapter 1 opening prose.
- *code* / *mixed_ja_en*: byte-identical excerpts of the upstream files.

Raw staged-file SHA-256 (as fetched 2026-09-03):

```
35f052e9b71a18e9189e61d3453587d2d0b210bcf7f6be36bc3b134650a92f40  code_cpython_dataclasses.py
2d9a76a2e3e8195c69430516ebd33c4d0757a53ad432ff6186b7b794e6fe99f9  general_en_gutenberg_walden.txt
d0124c6a71ca3c8f6898dc84578134789fcbdf6a1133121d205ebdaf52274594  kokoro/kokoro.txt (Shift-JIS)
fd30f07691f781737e409e63ac9a9f307600df1987c25c66befdfd17c800eb5b  mixed_ruby_readme_ja.md
```

The committed excerpt files are the fixture inputs of record; their SHA-256
values are recorded per case inside the fixture JSON.
