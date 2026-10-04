import hashlib
import json
import os
from datetime import datetime

LOG_FILE = 'audit_log.json'


def _hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def log_entry(result_dict, filepath):
    try:
        file_hash = _hash_file(filepath)
    except Exception:
        file_hash = "UNKNOWN"
    
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)
        prev_hash = logs[-1]['hash'] if logs else "GENESIS"
    else:
        logs = []
        prev_hash = "GENESIS"
    
    entry = {
        "timestamp": str(datetime.now()),
        "file": os.path.basename(filepath),
        "file_hash": file_hash,
        "result": result_dict,
        "prev_hash": prev_hash
    }
    
    entry_str = json.dumps(entry, sort_keys=True)
    entry['hash'] = hashlib.sha256(entry_str.encode()).hexdigest()
    
    logs.append(entry)
    
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    
    return entry['hash']


def verify_log_chain():
    if not os.path.exists(LOG_FILE):
        return "NO LOGS"
    
    with open(LOG_FILE, 'r') as f:
        logs = json.load(f)
    
    for i, entry in enumerate(logs):
        entry_copy = {k: v for k, v in entry.items() if k != 'hash'}
        entry_str = json.dumps(entry_copy, sort_keys=True)
        computed = hashlib.sha256(entry_str.encode()).hexdigest()
        
        if computed != entry['hash']:
            return f"❌ TAMPERED at entry {i}"
        
        if i > 0 and entry['prev_hash'] != logs[i-1]['hash']:
            return f"❌ CHAIN BROKEN at entry {i}"
    
    return f"✅ All {len(logs)} logs intact"


def get_logs(limit=100):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, 'r') as f:
        logs = json.load(f)
    return logs[-limit:]