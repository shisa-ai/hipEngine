#!/usr/bin/env python3
import argparse, collections, concurrent.futures, hashlib, json, pathlib, re, threading, time, urllib.request

IDS=("code_merge_intervals","code_topological_sort","code_lru_cache","code_markdown_table","general_en_plan","general_en_explain","general_ja_plan","general_ja_explain","mixed_ja_en_translate","mixed_ja_en_review")
def render(messages):
    out=[]
    for m in messages:
        role='system' if m['role']=='developer' else m['role']
        out.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>")
    out.append('<|im_start|>assistant\n')
    return '\n'.join(out)
def load(path):
    rows=[]
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip(): continue
        p=json.loads(line); text=render(p['messages']); rows.append({'id':p['id'],'category':p['category'],'heldout':p['id'].endswith(('markdown_table','_explain','_review')),'prompt':text,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()})
    assert tuple(x['id'] for x in rows)==IDS
    return rows
def post(url,payload,barrier):
    barrier.wait(30); started=time.perf_counter()
    req=urllib.request.Request(url+'/completion',json.dumps(payload).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as f: d=json.load(f)
    completed=time.perf_counter(); text=d.get('content') or ''; t=d.get('timings') or {}
    words=re.findall(r'\S+',text); tri=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]; windows=[text[i:i+30] for i in range(max(0,len(text)-29))]
    return {'started':started,'completed':completed,'wall_seconds':completed-started,'prompt_n':int(t.get('prompt_n') or 0),'prompt_ms':float(t.get('prompt_ms') or 0.0),'prompt_per_second':t.get('prompt_per_second'),'predicted_n':int(t.get('predicted_n') or 0),'predicted_per_second':t.get('predicted_per_second'),'draft_n':int(t.get('draft_n') or 0),'draft_n_accepted':int(t.get('draft_n_accepted') or 0),'content_sha256':hashlib.sha256(text.encode()).hexdigest(),'content':text,'unique_char30_fraction':len(set(windows))/len(windows) if windows else 1.0,'max_word_trigram_repeats':max(collections.Counter(tri).values(),default=0)}
def run(base,prompt,width,n):
    barrier=threading.Barrier(width+1); payload={'prompt':prompt,'n_predict':n,'temperature':0.0,'top_k':1,'cache_prompt':False}
    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as ex:
        fs=[ex.submit(post,base,payload,barrier) for _ in range(width)]; barrier.wait(30); rows=[f.result() for f in fs]
    wall=max(x['completed'] for x in rows)-min(x['started'] for x in rows); tokens=sum(x['predicted_n'] for x in rows)
    return {'wall_seconds':wall,'generated_tokens':tokens,'tok_s':tokens/wall,'rows':rows}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',required=True);ap.add_argument('--label',required=True);ap.add_argument('--arm',choices=('ar','mtp'),required=True);ap.add_argument('--prompts',required=True);ap.add_argument('--widths',default='1,2,3,4,5,6,7,8');ap.add_argument('--max-tokens',type=int,default=24);ap.add_argument('--model-sha256',required=True);ap.add_argument('--engine-commit',required=True);ap.add_argument('--backend',required=True);ap.add_argument('--server-command',required=True);ap.add_argument('--out',required=True);a=ap.parse_args(); prompts=load(a.prompts); widths=[int(x) for x in a.widths.split(',')]
    cells=[]
    for w in widths:
        run(a.base,prompts[0]['prompt'],w,a.max_tokens)
        for p in prompts:
            m=run(a.base,p['prompt'],w,a.max_tokens); cell={k:p[k] for k in ('id','category','heldout','prompt_sha256')};cell.update({'width':w,**m});cells.append(cell);print(json.dumps({'arm':a.arm,'width':w,'prompt':p['id'],'tok_s':m['tok_s']}),flush=True)
    summary={}
    for w in widths:
        rs=[x for x in cells if x['width']==w]; tokens=sum(x['generated_tokens'] for x in rs); wall=sum(x['wall_seconds'] for x in rs); prompt_tokens=sum(y['prompt_n'] for x in rs for y in x['rows']); prompt_wall_ms=sum(max((y['prompt_ms'] for y in x['rows']),default=0.0) for x in rs); drafted=sum(y['draft_n'] for x in rs for y in x['rows']); accepted=sum(y['draft_n_accepted'] for x in rs for y in x['rows']); summary[str(w)]={'prompts':len(rs),'requests':len(rs)*w,'prompt_tokens':prompt_tokens,'prompt_wall_ms':prompt_wall_ms,'aggregate_prompt_tok_s':prompt_tokens/(prompt_wall_ms/1000.0) if prompt_wall_ms else None,'generated_tokens':tokens,'wall_seconds':wall,'complete_wall_tok_s':tokens/wall,'draft_tokens':drafted,'draft_accepted':accepted,'acceptance':accepted/drafted if drafted else None,'min_unique_char30_fraction':min(y['unique_char30_fraction'] for x in rs for y in x['rows']),'max_word_trigram_repeats':max(y['max_word_trigram_repeats'] for x in rs for y in x['rows'])}
    out={'schema':1,'kind':'llamacpp_qwen38_standard_q4_c1c8','label':a.label,'arm':a.arm,'date':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'engine_commit':a.engine_commit,'backend':a.backend,'model_sha256':a.model_sha256,'protocol':{'prompts':a.prompts,'prompt_count':len(prompts),'widths':widths,'max_tokens':a.max_tokens,'temperature':0,'top_k':1,'cache_prompt':False,'timing':'barrier-to-last-completion complete wall per prompt; summed across prompts','server_command':a.server_command},'summary':summary,'cells':cells}
    pathlib.Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False));print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
