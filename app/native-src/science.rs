//! Scientific kernels with the existing FP32 masks and FP64 feature contracts.
use std::slice;

fn pixel(image:&[u8],h:usize,w:usize,r:isize,c:isize)->f64 {
    if r<0 || c<0 || r>=h as isize || c>=w as isize {0.0} else {image[r as usize*w+c as usize] as f64}
}
fn bilinear(image:&[u8],h:usize,w:usize,r:f64,c:f64)->f64 {
    let r0=r.floor() as isize;let r1=r.ceil() as isize;
    let c0=c.floor() as isize;let c1=c.ceil() as isize;
    let dr=r-r0 as f64;let dc=c-c0 as f64;
    let top=(1.0-dc)*pixel(image,h,w,r0,c0)+dc*pixel(image,h,w,r0,c1);
    let bottom=(1.0-dc)*pixel(image,h,w,r1,c0)+dc*pixel(image,h,w,r1,c1);
    (1.0-dr)*top+dr*bottom
}
fn lbp(image:&[u8],h:usize,w:usize,r:usize,c:usize)->u8 {
    let offsets=[(0.0,1.0),(-0.70711,0.70711),(-1.0,0.0),(-0.70711,-0.70711),
                 (0.0,-1.0),(0.70711,-0.70711),(1.0,0.0),(0.70711,0.70711)];
    let mut code=0u8;
    for (i,(dr,dc)) in offsets.iter().enumerate() {
        if bilinear(image,h,w,r as f64+dr,c as f64+dc)-image[r*w+c] as f64>=0.0 {code|=1<<i;}
    }
    code
}

#[no_mangle]
pub unsafe extern "C" fn hopbox_lbp(input:*const u8,h:usize,w:usize,output:*mut u8)->i32 {
    if input.is_null() || output.is_null() {return -1;}
    let Some(n)=h.checked_mul(w) else {return -2;};
    let src=slice::from_raw_parts(input,n);let dst=slice::from_raw_parts_mut(output,n);
    for r in 0..h {for c in 0..w {dst[r*w+c]=lbp(src,h,w,r,c);}}
    0
}

#[no_mangle]
pub unsafe extern "C" fn hopbox_channel_features(input:*const u8,h:usize,w:usize,output:*mut f64)->i32 {
    if input.is_null() || output.is_null() || h==0 || w==0 {return -1;}
    let Some(n)=h.checked_mul(w) else {return -2;};
    let src=slice::from_raw_parts(input,n);let out=slice::from_raw_parts_mut(output,30);
    let mut hist=[0u64;256];let mut lbps=[0u64;256];let mut max_code=0usize;
    for r in 0..h {for c in 0..w {
        hist[src[r*w+c] as usize]+=1;
        let code=lbp(src,h,w,r,c) as usize;lbps[code]+=1;max_code=max_code.max(code);
    }}
    // Preserve density=True histogram followed by the original normalization.
    let mut density=[0.0;256];let mut total_density=0.0;
    for i in 0..256 {density[i]=hist[i] as f64/(255.0/256.0)/n as f64;total_density+=density[i];}
    let mut uniformity=0.0;let mut entropy=0.0;
    for d in density {let p=d/total_density;uniformity+=p*p;if p>0.0 {entropy-=p*(p*p.log2());}}
    let mean_all=hist.iter().enumerate().map(|(i,&v)|i as f64*v as f64).sum::<f64>()/n as f64;
    let variance_all=hist.iter().enumerate().map(|(i,&v)|{let x=i as f64-mean_all;x*x*v as f64}).sum::<f64>()/n as f64;
    out[0]=1.0-1.0/(1.0+variance_all/(256.0*256.0));out[1]=uniformity;out[2]=entropy;
    let mut bins=[0u64;16];for (i,&count) in lbps.iter().enumerate() {bins[(i*16/(max_code+1)).min(15)]+=count;}
    for i in 0..16 {out[3+i]=bins[i] as f64/((max_code+1) as f64/16.0)/n as f64;}
    let count=n as u64-hist[0];
    if count==0 {for value in &mut out[19..24] {*value=f64::NAN;}}
    else {
        let mean=hist.iter().enumerate().skip(1).map(|(i,&v)|i as f64*v as f64).sum::<f64>()/count as f64;
        let mut m2=0.0;let mut m3=0.0;let mut m4=0.0;let mut cumulative=0u64;let mut low=0;let mut high=0;
        for (i,&v) in hist.iter().enumerate().skip(1) {
            let d=i as f64-mean;let d2=d*d;
            m2+=v as f64*d2;m3+=v as f64*(d2*d);m4+=v as f64*(d2*d2);
            if cumulative<=(count-1)/2 && cumulative+v>(count-1)/2 {low=i;}
            if cumulative<=count/2 && cumulative+v>count/2 {high=i;}
            cumulative+=v;
        }
        m2/=count as f64;m3/=count as f64;m4/=count as f64;
        out[19]=mean;out[20]=(low+high) as f64/2.0;out[21]=m2.sqrt();
        let zero=m2<=(f64::EPSILON*mean).powi(2);
        out[22]=if zero {f64::NAN} else {m3/m2.powf(1.5)};
        out[23]=if zero {f64::NAN} else {m4/(m2*m2)-3.0};
    }
    let mut props=[0.0;6];
    for (dr,dc) in [(0isize,1isize),(1,1),(1,0),(1,-1)] {
        let mut matrix=vec![0u64;256*256];let mut pairs=0u64;
        for r in 0..h {for c in 0..w {
            let rr=r as isize+dr;let cc=c as isize+dc;
            if rr>=0 && cc>=0 && rr<h as isize && cc<w as isize {
                matrix[src[r*w+c] as usize*256+src[rr as usize*w+cc as usize] as usize]+=1;pairs+=1;
            }
        }}
        let denom=pairs.max(1) as f64;let mut mean_i=0.0;let mut mean_j=0.0;
        let mut contrast=0.0;let mut dissimilarity=0.0;let mut homogeneity=0.0;let mut asm=0.0;
        for i in 0..256 {for j in 0..256 {
            let p=matrix[i*256+j] as f64/denom;let d=i as f64-j as f64;
            mean_i+=i as f64*p;mean_j+=j as f64*p;
            contrast+=p*d*d;dissimilarity+=p*d.abs();homogeneity+=p*(1.0/(1.0+d*d));asm+=p*p;
        }}
        let mut vi=0.0;let mut vj=0.0;let mut covariance=0.0;
        for i in 0..256 {for j in 0..256 {
            let p=matrix[i*256+j] as f64/denom;let di=i as f64-mean_i;let dj=j as f64-mean_j;
            vi+=p*(di*di);vj+=p*(dj*dj);covariance+=p*(di*dj);
        }}
        let correlation=if vi.sqrt()<1e-15 || vj.sqrt()<1e-15 {1.0} else {covariance/(vi.sqrt()*vj.sqrt())};
        for (a,v) in props.iter_mut().zip([contrast,dissimilarity,correlation,homogeneity,asm.sqrt(),asm]) {*a+=v/4.0;}
    }
    out[24..30].copy_from_slice(&props);
    0
}

#[no_mangle]
pub unsafe extern "C" fn hopbox_mask_roi(input:*const f32,ch:usize,cw:usize,h:usize,w:usize,
    x0:usize,y0:usize,x1:usize,y1:usize,output:*mut u8)->i32 {
    if input.is_null() || output.is_null() || ch==0 || cw==0 || h==0 || w==0 || x1<x0 || y1<y0 || x1>w || y1>h {return -1;}
    let Some(n)=ch.checked_mul(cw) else {return -2;};
    let Some(size)=(x1-x0).checked_mul(y1-y0) else {return -2;};
    let src=slice::from_raw_parts(input,n);let dst=slice::from_raw_parts_mut(output,size);
    let sx=(cw as f64/w as f64) as f32;let sy=(ch as f64/h as f64) as f32;
    let mut xs=Vec::with_capacity(x1-x0);
    for x in x0..x1 {
        let v=((x as f32+0.5)*sx-0.5).clamp(0.0,(cw-1) as f32);
        let a=v.floor() as usize;xs.push((a,(a+1).min(cw-1),v as f64-a as f64));
    }
    for y in y0..y1 {
        let v=((y as f32+0.5)*sy-0.5).clamp(0.0,(ch-1) as f32);
        let a=v.floor() as usize;let b=(a+1).min(ch-1);let wy=v as f64-a as f64;
        for (offset,&(xa,xb,wx)) in xs.iter().enumerate() {
            let top=src[a*cw+xa] as f64*(1.0-wx)+src[a*cw+xb] as f64*wx;
            let bottom=src[b*cw+xa] as f64*(1.0-wx)+src[b*cw+xb] as f64*wx;
            dst[(y-y0)*(x1-x0)+offset]=u8::from(top*(1.0-wy)+bottom*wy>0.0);
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn hopbox_closest_contours(a:*const i32,na:usize,b:*const i32,nb:usize,output:*mut usize)->i32 {
    if a.is_null() || b.is_null() || output.is_null() || na==0 || nb==0 {return -1;}
    let a=slice::from_raw_parts(a,na*2);let b=slice::from_raw_parts(b,nb*2);
    let mut best=i64::MAX;let mut pair=(0usize,0usize);
    for i in 0..na {for j in 0..nb {
        let dx=a[2*i] as i64-b[2*j] as i64;let dy=a[2*i+1] as i64-b[2*j+1] as i64;
        let distance=dx*dx+dy*dy;if distance<best {best=distance;pair=(i,j);}
    }}
    let out=slice::from_raw_parts_mut(output,2);out[0]=pair.0;out[1]=pair.1;0
}

#[no_mangle]
pub unsafe extern "C" fn hopbox_components(input:*const u8,h:usize,w:usize,output:*mut i32,
    stats:*mut i32,centroids:*mut f64,capacity:usize)->i32 {
    if input.is_null() || output.is_null() || stats.is_null() || centroids.is_null() || h==0 || w==0 {return -1;}
    let Some(n)=h.checked_mul(w) else {return -2;};
    let src=slice::from_raw_parts(input,n);let labels=slice::from_raw_parts_mut(output,n);labels.fill(0);
    let mut summaries:Vec<[i64;7]>=vec![[i64::MAX,i64::MAX,-1,-1,0,0,0]];
    let mut queue=Vec::<usize>::new();
    for p in 0..n {
        if src[p]==0 || labels[p]!=0 {continue;}
        if summaries.len()>=capacity {return -3;}
        let id=summaries.len() as i32;summaries.push([i64::MAX,i64::MAX,-1,-1,0,0,0]);
        labels[p]=id;queue.clear();queue.push(p);let mut head=0;
        while head<queue.len() {
            let q=queue[head];head+=1;let r=q/w;let col=q%w;
            for other in [if r>0 {Some(q-w)} else {None},if col>0 {Some(q-1)} else {None},
                          if col+1<w {Some(q+1)} else {None},if r+1<h {Some(q+w)} else {None}].into_iter().flatten() {
                if src[other]!=0 && labels[other]==0 {labels[other]=id;queue.push(other);}
            }
        }
    }
    for (p,&id) in labels.iter().enumerate() {
        let x=(p%w) as i64;let y=(p/w) as i64;let row=&mut summaries[id as usize];
        row[0]=row[0].min(x);row[1]=row[1].min(y);row[2]=row[2].max(x);row[3]=row[3].max(y);
        row[4]+=1;row[5]+=x;row[6]+=y;
    }
    let dst=slice::from_raw_parts_mut(stats,capacity*5);let centers=slice::from_raw_parts_mut(centroids,capacity*2);
    for (i,s) in summaries.iter().enumerate() {
        let values=if s[4]==0 {[-1,i32::MAX,0,0,0]} else {[s[0] as i32,s[1] as i32,(s[2]-s[0]+1) as i32,(s[3]-s[1]+1) as i32,s[4] as i32]};
        dst[i*5..i*5+5].copy_from_slice(&values);
        centers[i*2]=if s[4]==0 {f64::NAN} else {s[5] as f64/s[4] as f64};
        centers[i*2+1]=if s[4]==0 {f64::NAN} else {s[6] as f64/s[4] as f64};
    }
    summaries.len() as i32
}
