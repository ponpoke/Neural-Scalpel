from enum import Enum

class RouteStatus(Enum):
    DRAFT = "DRAFT"
    DIAGNOSED = "DIAGNOSED"
    STAGING = "STAGING"
    CANARY = "CANARY"
    PRODUCTION = "PRODUCTION"
    REVOKED = "REVOKED"
    QUARANTINED = "QUARANTINED"

class PolicyDecision(Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    MANUAL_REVIEW = "MANUAL_REVIEW"

def evaluate_license_risk(license_name: str) -> PolicyDecision:
    """
    Evaluates the risk of a given license for PRODUCTION deployment.
    """
    if not license_name:
        return PolicyDecision.MANUAL_REVIEW
        
    license_upper = license_name.upper()
    
    # Low risk permissive licenses
    if any(l in license_upper for l in ["MIT", "APACHE", "BSD"]):
        return PolicyDecision.ALLOW
        
    # High risk copyleft or restrictive licenses
    if any(l in license_upper for l in ["GPL", "AGPL", "CC-BY-NC-SA"]):
        return PolicyDecision.DENY
        
    # Unknown licenses
    return PolicyDecision.MANUAL_REVIEW
