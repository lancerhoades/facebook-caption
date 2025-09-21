# Usage: awk -v W=5 -v D=1.6 -f tools/srt_chunkify.awk in.srt > out.srt
function ts_to_s(t,   h,m,s,ms,a){gsub(",",".",t);split(t,a,":");h=a[1]+0;m=a[2]+0;split(a[3],a,".");s=a[1]+0;ms=(a[2]?a[2]+0:0);return h*3600+m*60+s+ms/1000.0}
function s_to_ts(x,   h,m,s,ms){if(x<0)x=0;h=int(x/3600);x-=h*3600;m=int(x/60);x-=m*60;s=int(x);ms=int((x-s)*1000+0.5);return sprintf("%02d:%02d:%02d,%03d",h,m,s,ms)}
function flush_block(    i,st,en,txt,words,n,parts,per,ci,cstart,cend,chunk,cw,a){
  if(tl=="")return; split(tl,a,/ *--> */); st=ts_to_s(a[1]); en=ts_to_s(a[2]); if(st>=en)return
  txt=""; for(i=1;i<=tln;i++){if(lines[i]~(/^[[:space:]]*$/))continue; if(txt!="")txt=txt"\n"; txt=txt lines[i]}
  n=split(txt,words,/[[:space:]]+/)
  if(W==0||W==""||n<=W){print ++global_idx; print s_to_ts(st)" --> "s_to_ts(en); print txt"\n"}
  else{
    parts=int((n+W-1)/W); per=(en-st)/parts
    for(ci=0;ci<parts;ci++){
      cstart=st+ci*per; cend=st+(ci+1)*per
      if(D>0 && cend-cstart>D) cend=cstart+D
      chunk=""; for(cw=ci*W+1;cw<=n && cw<=(ci+1)*W;cw++){if(chunk!="")chunk=chunk" "; chunk=chunk words[cw]}
      print ++global_idx; print s_to_ts(cstart)" --> "s_to_ts(cend); print chunk"\n"
    }
  }
  tl=""; tln=0; delete lines
}
BEGIN{RS="";FS="\n";OFS="\n";global_idx=0;W=(W?W:0);D=(D?D:0)}
{
  idx_seen=0; tl=""; tln=0
  for(i=1;i<=NF;i++){ if($i ~ /-->/ && tl==""){tl=$i; continue}
    if(tl=="") continue
    if(length($i)) lines[++tln]=$i
  }
  flush_block()
}
