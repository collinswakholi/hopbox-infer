//! Class-aware NMS with the application's float32 offsets and stable ties.
use std::{cmp::Ordering, slice};

#[no_mangle]
pub unsafe extern "C" fn hopbox_nms(
    boxes: *const f32, scores: *const f32, classes: *const i64,
    count: usize, threshold: f32, limit: usize, output: *mut i64,
) -> usize {
    if count == 0 || limit == 0 { return 0; }
    if boxes.is_null() || scores.is_null() || classes.is_null() || output.is_null() { return 0; }
    let boxes = slice::from_raw_parts(boxes, count * 4);
    let scores = slice::from_raw_parts(scores, count);
    let classes = slice::from_raw_parts(classes, count);
    let output = slice::from_raw_parts_mut(output, limit.min(count));
    let shifted: Vec<[f32; 4]> = (0..count).map(|i| {
        let offset = classes[i] as f32 * 7680.0;
        [boxes[4*i]+offset, boxes[4*i+1]+offset, boxes[4*i+2]+offset, boxes[4*i+3]+offset]
    }).collect();
    let areas: Vec<f32> = shifted.iter().map(|b| (b[2]-b[0]).max(0.0)*(b[3]-b[1]).max(0.0)).collect();
    let mut order: Vec<usize> = (0..count).collect();
    // Stable sort retains the original order for equal confidence scores.
    order.sort_by(|&a,&b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
    order.truncate(30000);
    let mut kept = 0;
    while !order.is_empty() && kept < output.len() {
        let i = order[0]; output[kept] = i as i64; kept += 1;
        order.remove(0);
        order.retain(|&j| {
            let x = (shifted[i][2].min(shifted[j][2])-shifted[i][0].max(shifted[j][0])).max(0.0);
            let y = (shifted[i][3].min(shifted[j][3])-shifted[i][1].max(shifted[j][1])).max(0.0);
            let intersection = x*y;
            let union = areas[i]+areas[j]-intersection;
            let overlap = if union > 0.0 { intersection/union } else { 0.0 };
            overlap <= threshold
        });
    }
    kept
}
