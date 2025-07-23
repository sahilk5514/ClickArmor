import json

# Feature configuration for Training vs Inference
feature_config = {
    "common_features": [
        "url_length", "num_digits", "num_special_chars",
        "entropy_url", "entropy_domain", "entropy_filename",
        "suspicious_keywords", "num_subdomains", "path_length",
        "tld_flag", "homoglyph_flag"
    ],
    "training_only_features": [
        "has_valid_ssl", "domain_age", "whois_org", "ssl_issuer", "crtsh_verified"
    ],
    "inference_only_features": [
        "cached_ssl_flag", "cached_domain_age", "tranco_rank", "precomputed_ssl"  # precomputed
    ],
    "async_capable_features": [
        "has_valid_ssl", "domain_age", "crtsh_verified", "ssl_issuer"
    ],
    "fallback_defaults": {
        "has_valid_ssl": -1,
        "domain_age": -1,
        "crtsh_verified": 0,
        "ssl_issuer": "unknown"
    }
}

# Save config
with open("feature_config.json", "w") as f:
    json.dump(feature_config, f, indent=4)

print("Feature config saved as feature_config.json")
