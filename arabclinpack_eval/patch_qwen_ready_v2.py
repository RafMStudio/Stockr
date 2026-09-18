from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text(encoding="utf8")
s=s.replace("SEED=17\n", "TARGET_REDUCTION=float(os.environ.get('TARGET_REDUCTION','0.30'))\nSEED=17\n")
old="""        sel=self.candidates(text,ids,offs)
        base=torch.tensor(ids,dtype=torch.long)
"""
new="""        sel=self.candidates(text,ids,offs)
        # Safety/quality budget: do not compress beyond the requested Qwen-native target.
        # Keep semantically richer domain spans first, then high-saving word merges.
        target_save=max(0,int(round(len(ids)*TARGET_REDUCTION)))
        if sum(x['save'] for x in sel)>target_save:
            chosen=[]; used=0
            for x in sorted(sel,key=lambda z:(-z['bonus'],-z['save'],z['ts'])):
                if used+x['save']<=target_save:
                    chosen.append(x); used+=x['save']
            sel=sorted(chosen,key=lambda z:z['ts'])
        base=torch.tensor(ids,dtype=torch.long)
"""
if old not in s: raise SystemExit("selection patch target missing")
s=s.replace(old,new)
s=s.replace("text=train[(step*37)%len(train)] + '\\nالجواب:'", "text=train[(step*37)%len(train)]")
old2="""        kl=F.kl_div(F.log_softmax(slog/2.0,dim=-1),tp,reduction='batchmean')*4.0
        cos=1-F.cosine_similarity(sh,th,dim=0)
        mse=F.mse_loss(F.layer_norm(sh,sh.shape),F.layer_norm(th,th.shape))
        loss=cos + 0.15*mse + 0.01*kl
"""
new2="""        # Proper full-distribution distillation. 'batchmean' on a 1-D vocabulary vector
        # dilutes KL by vocab size, so use sum and add direct teacher-top1 supervision.
        kl=F.kl_div(F.log_softmax(slog/2.0,dim=-1),tp,reduction='sum')*4.0
        ce=F.cross_entropy(slog[None,:],tlog.argmax().view(1))
        cos=1-F.cosine_similarity(sh,th,dim=0)
        mse=F.mse_loss(F.layer_norm(sh,sh.shape),F.layer_norm(th,th.shape))
        loss=cos + 0.05*mse + 0.05*kl + 0.10*ce
"""
if old2 not in s: raise SystemExit("loss patch target missing")
s=s.replace(old2,new2)
s=s.replace("logs.append({'step':step,'loss':float(loss),'cos':float(cos),'mse':float(mse),'kl':float(kl)})", "logs.append({'step':step,'loss':float(loss),'cos':float(cos),'mse':float(mse),'kl':float(kl),'ce':float(ce)})")
p.write_text(s,encoding="utf8")
