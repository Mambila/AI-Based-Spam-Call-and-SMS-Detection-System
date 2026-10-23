def score_call_metadata(duration_seconds, frequency_today, international_code):
    """
    Evaluates call risk based on anonymised metadata (duration, frequency, time of day).
    Returns: Risk tier ('Safe', 'Suspicious', or 'Spam') and numeric score.
    """
    risk_score = 0
    
    # Very short calls can indicate robocalls / missed call scams (pings)
    if duration_seconds < 3:
        risk_score += 2
    
    # High frequency of calls from the same number in one day
    if frequency_today > 5:
        risk_score += 3
        
    # Unfamiliar or suspicious international formatting flags
    if international_code:
        risk_score += 1
        
    # Map score to risk tiers
    if risk_score >= 4:
        return "Spam", risk_score
    elif risk_score >= 2:
        return "Suspicious", risk_score
    else:
        return "Safe", risk_score