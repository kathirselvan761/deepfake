def fuse_scores(visual_fake, audio_fake=0.0, lip_sync_mismatch=None):
    """Multi-modal scores fuse panni final verdict kudu."""
    visual_fake = max(0.0, min(1.0, visual_fake))
    audio_fake = max(0.0, min(1.0, audio_fake))
    
    if lip_sync_mismatch is not None:
        lip_sync_mismatch = max(0.0, min(1.0, lip_sync_mismatch))
        final = 0.5 * visual_fake + 0.3 * audio_fake + 0.2 * lip_sync_mismatch
        breakdown = {
            'visual': round(visual_fake, 4),
            'audio': round(audio_fake, 4),
            'lip_sync': round(lip_sync_mismatch, 4)
        }
    else:
        final = 0.7 * visual_fake + 0.3 * audio_fake
        breakdown = {
            'visual': round(visual_fake, 4),
            'audio': round(audio_fake, 4),
            'lip_sync': None
        }
    
    if final > 0.7:
        verdict = "DEEPFAKE"
    elif final > 0.4:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LIKELY REAL"
    
    trust_score = round((1 - final) * 100, 2)
    confidence = round(abs(final - 0.5) * 2 * 100, 2)
    
    if verdict == "DEEPFAKE":
        explanation = f"Strong manipulation evidence. Visual: {visual_fake*100:.1f}%"
    elif verdict == "SUSPICIOUS":
        explanation = "Some manipulation signals. Manual review recommended."
    else:
        explanation = "Media appears authentic."
    
    return {
        'verdict': verdict,
        'trust_score': trust_score,
        'confidence': confidence,
        'final_fake_score': round(final, 4),
        'breakdown': breakdown,
        'explanation': explanation
    }