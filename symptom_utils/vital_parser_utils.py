def parse_age(age):
    try:
        return float(age)
    except:
        return 92

def bucketize_age(age):
    if age < 18:
        return 'CHILD'
    elif age < 34:
        return '18_33'
    elif age < 49:
        return '34_48'
    elif age < 65:
        return '48_64'
    elif age < 78:
        return '64_77'
    return '78+'

def parse_temp(temp):
    if temp is None:
        return None
    temp = temp.strip()
    try:
        return float(temp)
    except:
        return None

def bucketize_temp(temp):
    if temp > 100.4:
        return 'HIGH_TEMP'
    elif temp < 97:
        return 'LOW_TEMP'
    else:
        return 'NORMAL_TEMP'

def parse_resp_rate(rr):
    if rr is None:
        return None
    rr = rr.strip()
    if not rr.isdigit():
        return None
    rr = int(float(rr))
    return rr

def bucketize_resp_rate(rr):
     if rr < 12:
         return 'LOW_RR'
     elif rr < 21:
         return 'NORMAL_RR'
     else:
         return 'HIGH_RR'

def parse_pulse_ox(sao2):
    if sao2 is None:
        return None
    sao2 = sao2.strip()
    try:
        if sao2.isdigit():
            return float(sao2)
        elif '%' in sao2:
            return float(sao2.split('%')[0].strip())
        elif sao2[:3].isdigit():
            return float(sao2[:3])
        elif sao2[:2].isdigit():
            return float(sao2[:2])
        return None
    except:
        return None

def bucketize_pulse_ox(sao2):
    return 'NORMAL_SAO2' if sao2 >= 95 else 'LOW_SAO2'

def parse_heart_rate(hr):
    if hr is None or not hr.isdigit():
        return None
    return float(hr) if float(hr) < 400 else None

def bucketize_heart_rate(hr):
    if hr > 100:
        return 'TACHYCARDIC_HR'
    if hr < 60:
        return 'BRADYCARDIC_HR'
    return 'NORMAL_HR'

def parse_blood_pressure(bp):
    if bp is None or '/' not in bp:
        return None, None
    systolic_bp = bp.split('/')[0]
    diastolic_bp = bp.split('/')[1]
    try:
        systolic_bp = float(systolic_bp)
        diastolic_bp = float(diastolic_bp)
        return systolic_bp, diastolic_bp
    except:
        return None, None

def bucketize_blood_pressure(systolic_bp, diastolic_bp):
    if systolic_bp < 120 and diastolic_bp < 80:
        return 'NORMAL_BP'
    elif systolic_bp < 130 and diastolic_bp < 80:
        return 'ELEVATED_BP'
    elif systolic_bp < 140 and diastolic_bp < 90:
        return 'STAGE1_HYPERTENSION'
    return 'STAGE2_HYPERTENSION'
