//! Compiled FP32 color application. No ML framework or compiler at runtime.
use std::slice;
const LUT: usize = 33*33*33*3;
const PARAMETERS: usize = LUT+9+9+3+3+8*4;

#[inline] fn linear(x:f32)->f32 { if x<=0.04045 {x/12.92} else {((x+0.055)/1.055).powf(2.4)} }
#[inline] fn srgb(x:f32)->f32 { if x<=0.0031308 {x*12.92} else {1.055*x.powf(1.0/2.4)-0.055} }
#[inline] fn lab(x:f32)->f32 { if x>216.0/24389.0 {x.powf(1.0/3.0)} else {(24389.0/27.0*x+16.0)/116.0} }
#[inline] fn inv_lab(x:f32)->f32 {if x>6.0/29.0 {x*x*x} else {(116.0*x-16.0)/(24389.0/27.0)}}
#[inline] fn dot(x:[f32;3],m:&[f32],row:usize)->f32 {x[0]*m[row*3]+x[1]*m[row*3+1]+x[2]*m[row*3+2]}

fn map_lightness(x:f32,p:&[f32],spline:bool,n:usize,count:usize)->f32 {
    let x=x.clamp(0.0,100.0);
    let knots=&p[LUT+24..LUT+32];let values=&p[LUT+32..LUT+40];let slopes=&p[LUT+40..LUT+48];
    if spline {
        if x<=knots[0] {return values[0]}
        if x>=knots[n-1] {return values[n-1]}
        let upper=(1..n).find(|&i|knots[i]>x).unwrap_or(n-1);
        let lower=upper-1;let span=(knots[upper]-knots[lower]).max(f32::EPSILON);
        let t=(x-knots[lower])/span;let t2=t*t;let t3=t2*t;
        let h00=2.0*t3-3.0*t2+1.0;let h10=t3-2.0*t2+t;let h01=-2.0*t3+3.0*t2;let h11=t3-t2;
        return h00*values[lower]+h10*span*slopes[lower]+h01*values[upper]+h11*span*slopes[upper]
    }
    let mut value=0.0;
    for &coefficient in &p[LUT+48..LUT+48+count] {value=value*x+coefficient}
    value
}

fn gamma(rgb:[f32;3],p:&[f32],spline:bool,n:usize,count:usize)->[f32;3] {
    let rgb=[linear(rgb[0]),linear(rgb[1]),linear(rgb[2])];
    let matrix=&p[LUT..LUT+9];let inverse=&p[LUT+9..LUT+18];let white=&p[LUT+18..LUT+21];
    let fx=lab(dot(rgb,matrix,0)/white[0]);let fy=lab(dot(rgb,matrix,1)/white[1]);let fz=lab(dot(rgb,matrix,2)/white[2]);
    let mapped=map_lightness(116.0*fy-16.0,p,spline,n,count).clamp(0.0,100.0);
    let a=500.0*(fx-fy);let b=200.0*(fy-fz);let fy2=(mapped+16.0)/116.0;
    let xyz=[inv_lab(a/500.0+fy2)*white[0],inv_lab(fy2)*white[1],inv_lab(fy2-b/200.0)*white[2]];
    [srgb(dot(xyz,inverse,0).clamp(0.0,1.0)).clamp(0.0,1.0),
     srgb(dot(xyz,inverse,1).clamp(0.0,1.0)).clamp(0.0,1.0),
     srgb(dot(xyz,inverse,2).clamp(0.0,1.0)).clamp(0.0,1.0)]
}

fn sample(rgb:[f32;3],p:&[f32])->[f32;3] {
    let mut lo=[0usize;3];let mut t=[0.0f32;3];
    for i in 0..3 {
        let x=rgb[i].clamp(0.0,1.0)*32.0;
        lo[i]=x as usize;
        if lo[i]>=32 {lo[i]=31;t[i]=1.0} else {t[i]=x-lo[i] as f32}
    }
    let (r,g,b)=(lo[0],lo[1],lo[2]);let (tr,tg,tb)=(t[0],t[1],t[2]);
    let (nr,ng,nb)=(1.0-tr,1.0-tg,1.0-tb);
    let entries=[(r,g,b,nr*ng*nb),(r+1,g,b,tr*ng*nb),(r,g+1,b,nr*tg*nb),(r+1,g+1,b,tr*tg*nb),
                 (r,g,b+1,nr*ng*tb),(r+1,g,b+1,tr*ng*tb),(r,g+1,b+1,nr*tg*tb),(r+1,g+1,b+1,tr*tg*tb)];
    let mut out=[0.0f32;3];
    for (r,g,b,weight) in entries {for c in 0..3 {out[c]+=weight*p[((r*33+g)*33+b)*3+c]}}
    out.map(|x|x.clamp(0.0,1.0))
}

#[no_mangle]
pub unsafe extern "C" fn hopbox_color_cpu(input:*const u8,output:*mut u8,pixels:usize,
    parameters:*const f32,parameter_count:usize,gc:i32,wb:i32,cc:i32,spline:i32,knot_n:i32,coeff_n:i32)->i32 {
    if input.is_null()||output.is_null()||parameters.is_null()||parameter_count!=PARAMETERS||!(0..=8).contains(&knot_n)||!(0..=8).contains(&coeff_n)||(spline!=0&&knot_n<2) {return -1}
    let Some(bytes)=pixels.checked_mul(3) else {return -2};
    let src=slice::from_raw_parts(input,bytes);let dst=slice::from_raw_parts_mut(output,bytes);
    let p=slice::from_raw_parts(parameters,parameter_count);
    let threads=std::thread::available_parallelism().map(|n|n.get()).unwrap_or(1).min(24);
    let chunk=pixels.div_ceil(threads).max(1)*3;
    std::thread::scope(|scope| {
        for (src,dst) in src.chunks(chunk).zip(dst.chunks_mut(chunk)) {
            scope.spawn(move || {
                for (a,b) in src.chunks_exact(3).zip(dst.chunks_exact_mut(3)) {
                    let mut rgb=[a[2] as f32*(1.0/255.0),a[1] as f32*(1.0/255.0),a[0] as f32*(1.0/255.0)];
                    if gc!=0 {rgb=gamma(rgb,p,spline!=0,knot_n as usize,coeff_n as usize)}
                    if wb!=0 {for i in 0..3 {rgb[i]=(rgb[i]*p[LUT+21+i]).clamp(0.0,1.0)}}
                    if cc!=0 {rgb=sample(rgb,p)}
                    for i in 0..3 {b[i]=(rgb[2-i]*255.0+0.5).clamp(0.0,255.0) as u8}
                }
            });
        }
    });
    0
}
