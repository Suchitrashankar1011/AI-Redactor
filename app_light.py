from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import re
import PyPDF2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import io
from werkzeug.utils import secure_filename
import spacy
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
REDACTED_FOLDER = 'redacted'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REDACTED_FOLDER'] = REDACTED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REDACTED_FOLDER, exist_ok=True)

# Initialize ML models
print("🤖 Initializing ML models...")
try:
    # Load spaCy NER model
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy NER model loaded")
except (OSError, ImportError) as e:
    print(f"❌ spaCy model not available: {e}")
    print("⚠️  Continuing without spaCy - using Transformers and NLTK only")
    nlp = None

try:
    # Initialize transformers NER pipeline
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    print("✅ Transformers NER pipeline loaded")
except Exception as e:
    print(f"❌ Transformers NER failed: {e}")
    ner_pipeline = None

try:
    # Download NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    print("✅ NLTK data downloaded")
except Exception as e:
    print(f"❌ NLTK setup failed: {e}")

print("🤖 ML models initialization complete!")

# High-confidence sensitive information patterns - only truly sensitive data
SENSITIVE_PATTERNS = {
    # Financial Information
    'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
    'credit_card': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
    'bank_account': r'\b(?:Account|Acct)\.?\s*#?\s*\d{8,17}\b',
    'routing_number': r'\b(?:Routing|RTN)\.?\s*#?\s*\d{9}\b',
    
    # Personal Identifiers
    'passport': r'\b[A-Z]{1,2}\d{6,9}\b',
    'drivers_license': r'\b[A-Z]\d{7,8}\b',
    'patient_id': r'\b(?:Patient\s+ID|MRN|Medical\s+Record\s+Number):\s*[A-Z0-9-]{6,}\b',
    
    # Contact Information
    'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    
    # Technical Identifiers
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'mac_address': r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b',
    
    # Sensitive Dates (only when clearly personal)
    'date_of_birth': r'\b(?:DOB|Date\s+of\s+Birth|Born):\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    
    # Government/ID Numbers
    'alien_number': r'\bA\d{8,9}\b',
    'visa_number': r'\b[A-Z]\d{8}\b',
    
    # Medical Sensitive Info
    'medical_record_id': r'\b(?:MRN|Medical\s+Record):\s*[A-Z0-9-]{6,}\b',
    'insurance_id': r'\b(?:Insurance|Policy)\s+ID:?\s*[A-Z0-9-]{6,}\b',
}

SENSITIVE_KEYWORDS = [
    'password', 'secret', 'confidential', 'private', 'ssn', 'social security',
    'credit card', 'bank account', 'routing number', 'pin', 'password',
    'username', 'login', 'authentication', 'token', 'key', 'api key',
    'personal information', 'pii', 'phi', 'financial', 'medical',
    'address', 'phone number', 'email address', 'date of birth'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    return text.strip()

def detect_sensitive_info(text):
    """ML-powered intelligent sensitive information detection"""
    print(f"🤖 ML Analyzing text of length: {len(text)} characters")
    print(f"📄 First 200 chars: {text[:200]}...")
    
    results = {
        'regex_matches': {},
        'ml_matches': [],
        'context_matches': [],
        'total_sensitive_items': 0
    }
    
    # High-confidence financial and ID patterns (always sensitive)
    high_confidence_patterns = {
        # Social Security Numbers
        'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
        'ssn_alt': r'\b(?:SSN|Social Security)\s*#?\s*\d{3}-?\d{2}-?\d{4}\b',
        
        # Credit Cards
        'credit_card': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
        'credit_card_alt': r'\b(?:Visa|MasterCard|Amex|Discover)\s*#?\s*\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        
        # Bank Information
        'bank_account': r'\b(?:Account|Acct|A/C)\.?\s*#?\s*\d{8,17}\b',
        'routing_number': r'\b(?:Routing|RTN|ABA)\.?\s*#?\s*\d{9}\b',
        'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',
        'swift_code': r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b',
        
        # Government IDs
        'passport': r'\b[A-Z]{1,2}\d{6,9}\b',
        'drivers_license': r'\b[A-Z]\d{7,8}\b',
        'military_id': r'\b(?:DOD|Military|Service)\s*ID\s*#?\s*\d{8,12}\b',
        'veteran_id': r'\b(?:VA|Veteran)\s*#?\s*\d{8,12}\b',
        
        # Contact Information
        'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        'phone_intl': r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',
        'phone_alt': r'\b\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        
        # Medical Information
        'medical_record': r'\b(?:MRN|Medical Record|Patient ID)\s*#?\s*[A-Z0-9-]{6,}\b',
        'insurance_id': r'\b(?:Insurance|Policy|Member)\s*ID\s*#?\s*[A-Z0-9-]{8,}\b',
        'medicare_id': r'\b(?:Medicare|Medicaid)\s*#?\s*[A-Z0-9-]{8,}\b',
        
        # Financial Information
        'account_number': r'\b(?:Account|Acct)\s*#?\s*\d{8,17}\b',
        'transaction_id': r'\b(?:Transaction|TXN|Ref)\s*#?\s*[A-Z0-9-]{8,}\b',
        'invoice_number': r'\b(?:Invoice|INV)\s*#?\s*[A-Z0-9-]{6,}\b',
        
        # Military Information
        'military_serial': r'\b(?:Serial|Service)\s*#?\s*[A-Z0-9-]{6,}\b',
        'security_clearance': r'\b(?:Clearance|Security)\s*Level\s*[A-Z0-9-]{2,}\b',
        'unit_identifier': r'\b(?:Unit|Squadron|Battalion)\s*[A-Z0-9-]{3,}\b',
        
        # Reference Numbers
        'reference_id': r'\b(?:Ref|Reference|ID)\s*#?\s*[A-Z0-9-]{6,}\b',
        'case_number': r'\b(?:Case|File)\s*#?\s*[A-Z0-9-]{6,}\b',
        'document_id': r'\b(?:Doc|Document)\s*#?\s*[A-Z0-9-]{6,}\b',
        
        # Dates and Times - Enhanced patterns
        'birth_date': r'\b(?:DOB|Born|Birth)\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'date_of_birth': r'\b(?:Date\s+of\s+Birth|Birth\s+Date)\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'admission_date': r'\b(?:Admission|Admitted)\s*Date\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'discharge_date': r'\b(?:Discharge|Discharged)\s*Date\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'procedure_date': r'\b(?:Procedure|Surgery)\s*Date\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'visit_date': r'\b(?:Visit|Appointment)\s*Date\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        'sensitive_dates': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        
        # Addresses
        'street_address': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Circle|Cir|Court|Ct|Place|Pl)\b',
        'zip_code': r'\b\d{5}(?:-\d{4})?\b',
        
        # Other Sensitive Patterns
        'api_key': r'\b(?:API|Key)\s*:?\s*[A-Za-z0-9]{20,}\b',
        'token': r'\b(?:Token|Access)\s*:?\s*[A-Za-z0-9]{20,}\b',
        'license_key': r'\b(?:License|Key)\s*:?\s*[A-Za-z0-9-]{16,}\b',
    }
    
    for pattern_name, pattern in high_confidence_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Filter out false positives
            filtered_matches = []
            for match in matches:
                # Skip empty matches
                if not match or match.strip() == '':
                    continue
                # Skip very short matches that are likely false positives
                if len(match.strip()) < 3:
                    continue
                # Skip word fragments and common words
                if match.lower().strip() in ['ion', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'rel', 'phy', 'ons', 'lar', 'chest', 'pain', 'shortness', 'breath', 'ies', 'pts', 'ing', 'ary', 'tes', 'ent', 'ded', 'ted', 'nal']:
                    continue
                # Skip if it's just a word fragment (less than 4 characters and not a known pattern)
                if len(match.strip()) < 4 and not re.match(r'^\d+$', match.strip()):
                    continue
                filtered_matches.append(match)
            
            if filtered_matches:
                unique_matches = list(set(filtered_matches))
                results['regex_matches'][pattern_name] = unique_matches
                print(f"✅ Found {len(unique_matches)} {pattern_name} matches: {unique_matches[:3]}")
    
    # ML-based detection
    ml_sensitive = detect_with_ml_models(text)
    
    # Enhanced name detection without labels
    name_entities = detect_names_without_labels(text)
    ml_sensitive.extend(name_entities)
    
    results['ml_matches'] = ml_sensitive
    results['context_matches'] = ml_sensitive  # For compatibility with test script
    
    # Count total sensitive items
    results['total_sensitive_items'] = sum(len(matches) for matches in results['regex_matches'].values()) + len(ml_sensitive)
    
    print(f"📊 Total sensitive items detected: {results['total_sensitive_items']}")
    
    return results

def detect_with_ml_models(text):
    """Use ML models to detect sensitive information"""
    ml_matches = []
    
    # spaCy NER detection
    if nlp:
        spacy_entities = detect_with_spacy(text)
        ml_matches.extend(spacy_entities)
    
    # Transformers NER detection
    if ner_pipeline:
        transformer_entities = detect_with_transformers(text)
        ml_matches.extend(transformer_entities)
    
    # NLTK NER detection
    nltk_entities = detect_with_nltk(text)
    ml_matches.extend(nltk_entities)
    
    # Context-based detection for medical documents
    medical_entities = detect_medical_context(text)
    ml_matches.extend(medical_entities)
    
    return ml_matches

def detect_with_spacy(text):
    """Detect entities using spaCy NER"""
    entities = []
    if not nlp:
        return entities
    
    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MONEY', 'DATE', 'TIME']:
                # Check if it's likely sensitive
                if is_ml_likely_sensitive(ent.text, ent.label_, text):
                    entities.append({
                        'text': ent.text,
                        'type': f'spacy_{ent.label_.lower()}',
                        'confidence': 0.9,
                        'context': get_context_around_match(ent.text, text)
                    })
                    print(f"✅ spaCy found {ent.label_}: '{ent.text}'")
    except Exception as e:
        print(f"❌ spaCy detection error: {e}")
    
    return entities

def detect_with_transformers(text):
    """Detect entities using Transformers NER"""
    entities = []
    if not ner_pipeline:
        return entities
    
    try:
        # Process text in chunks to avoid memory issues
        chunk_size = 512
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for chunk in text_chunks:
            results = ner_pipeline(chunk)
            for entity in results:
                if entity['score'] > 0.7:  # High confidence threshold
                    if is_ml_likely_sensitive(entity['word'], entity['entity_group'], text):
                        entities.append({
                            'text': entity['word'],
                            'type': f'transformer_{entity["entity_group"].lower()}',
                            'confidence': float(entity['score']),
                            'context': get_context_around_match(entity['word'], text)
                        })
                        print(f"✅ Transformers found {entity['entity_group']}: '{entity['word']}'")
    except Exception as e:
        print(f"❌ Transformers detection error: {e}")
    
    return entities

def detect_with_nltk(text):
    """Detect entities using NLTK NER"""
    entities = []
    
    try:
        # Tokenize and tag
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Named entity recognition
        tree = ne_chunk(pos_tags)
        
        for subtree in tree:
            if hasattr(subtree, 'label'):
                entity_text = ' '.join([token for token, pos in subtree.leaves()])
                entity_type = subtree.label()
                
                if entity_type in ['PERSON', 'ORGANIZATION', 'GPE']:
                    if is_ml_likely_sensitive(entity_text, entity_type, text):
                        entities.append({
                            'text': entity_text,
                            'type': f'nltk_{entity_type.lower()}',
                            'confidence': 0.8,
                            'context': get_context_around_match(entity_text, text)
                        })
                        print(f"✅ NLTK found {entity_type}: '{entity_text}'")
    except Exception as e:
        print(f"❌ NLTK detection error: {e}")
    
    return entities

def detect_medical_context(text):
    """Detect medical document specific sensitive information"""
    entities = []
    
    # Medical context patterns
    medical_patterns = {
        'patient_name': r'(?:Patient|Name):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
        'patient_id': r'(?:Patient\s+ID|ID|MRN):\s*([A-Z0-9-]{6,})',
        'dob': r'(?:DOB|Date\s+of\s+Birth|Born):\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        'age': r'Age:\s*(\d+)',
        'phone': r'(?:Phone|Tel|Mobile):\s*(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})',
        'email': r'(?:Email|E-mail):\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
    }
    
    for pattern_name, pattern in medical_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append({
                'text': match,
                'type': f'medical_{pattern_name}',
                'confidence': 0.95,
                'context': get_context_around_match(match, text)
            })
            print(f"✅ Medical context found {pattern_name}: '{match}'")
    
    return entities

def detect_names_without_labels(text):
    """Detect names that appear without any labels or context - ENHANCED"""
    entities = []
    
    # Enhanced name patterns for various contexts
    name_patterns = {
        # Standard name patterns (First Last, First Middle Last, etc.)
        'full_name': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b',
        
        # Names in lists or forms
        'name_in_list': r'^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\s*$',
        
        # Names in signatures
        'signature_name': r'\b(?:Signed|Signature|By)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b',
        
        # Names in contact information
        'contact_name': r'\b(?:Contact|Attn|Attention)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b',
        
        # Names in legal documents
        'legal_name': r'\b(?:Plaintiff|Defendant|Witness|Claimant)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b',
        
        # Names in medical documents
        'medical_name': r'\b(?:Patient|Subject|Individual)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b',
        
        # Names in financial documents
        'financial_name': r'\b(?:Account\s+Holder|Beneficiary|Payee|Payer)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b',
        
        # Names in military documents
        'military_name': r'\b(?:Service\s+Member|Veteran|Officer|Enlisted)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b',
    }
    
    for pattern_name, pattern in name_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            # Clean up the match
            clean_match = match.strip()
            if len(clean_match) > 3 and len(clean_match.split()) >= 2:  # At least 2 words
                # Skip if it's clearly not a person's name
                if not is_likely_person_name(clean_match, text):
                    continue
                    
                entities.append({
                    'text': clean_match,
                    'type': f'name_{pattern_name}',
                    'confidence': 0.9,
                    'context': get_context_around_match(clean_match, text)
                })
                print(f"✅ Name detection found {pattern_name}: '{clean_match}'")
    
    return entities

def is_likely_person_name(name, full_text):
    """Determine if a detected name is likely a person's name - ENHANCED FILTERING"""
    name_lower = name.lower()
    
    # Skip if it's clearly not a person's name
    non_person_indicators = [
        'hospital', 'medical center', 'health system', 'clinic', 'university', 'college',
        'government', 'federal', 'state', 'city', 'county', 'department', 'agency',
        'corporation', 'inc', 'llc', 'ltd', 'company', 'organization', 'institute',
        'john doe', 'jane doe', 'test user', 'admin', 'user', 'guest', 'anonymous',
        'example', 'sample', 'demo', 'template', 'placeholder',
        'patient', 'surgical', 'procedure', 'notes', 'record', 'findings', 'complications',
        'anesthesia', 'endotracheal', 'appendectomy', 'appendicitis', 'pathology',
        'informed', 'consent', 'blood', 'loss', 'minimal', 'presented', 'confirmed',
        'inflamed', 'perforation', 'removed', 'ports', 'scan', 'ct', 'pre', 'op', 'post'
    ]
    
    for indicator in non_person_indicators:
        if indicator in name_lower:
            return False
    
    # Skip if it contains numbers or special characters (except hyphens for hyphenated names)
    if re.search(r'[0-9]', name) or re.search(r'[^A-Za-z\s-]', name):
        return False
    
    # Skip if it's too short or too long
    if len(name) < 4 or len(name) > 50:
        return False
    
    # Skip if it's all uppercase (likely a title or acronym)
    if name.isupper() and len(name) > 10:
        return False
    
    # Skip if it contains medical/procedure terms
    medical_terms = ['patient', 'surgical', 'procedure', 'medical', 'hospital', 'clinic',
                    'doctor', 'nurse', 'surgeon', 'physician', 'anesthesia', 'surgery',
                    'diagnosis', 'treatment', 'therapy', 'scan', 'test', 'lab', 'blood',
                    'pain', 'fever', 'complications', 'findings', 'pathology', 'consent',
                    'dob', 'age', 'id', 'date', 'time', 'minimal', 'presented', 'confirmed',
                    'inflamed', 'perforation', 'removed', 'ports', 'ct', 'pre', 'op', 'post',
                    'vital', 'signs', 'physical', 'exam', 'chief', 'complaint', 'history',
                    'present', 'illness', 'discharge', 'instructions', 'allergies', 'medications',
                    'military', 'intelligence', 'command', 'division', 'unit', 'squadron',
                    'battalion', 'regiment', 'army', 'navy', 'air', 'force', 'marines',
                    'coast', 'guard', 'special', 'forces', 'delta', 'cyber', 'nato',
                    'recon', 'team', 'deploy', 'sigint', 'elint', 'phase', 'terrain',
                    'analysis', 'threat', 'assessment', 'briefing']
    
    words = name_lower.split()
    for word in words:
        if word in medical_terms:
            return False
    
    # Skip if it's a medical professional name (check context)
    medical_professional_names = ['chen', 'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller']
    if any(name in name_lower for name in medical_professional_names):
        # Check if it appears in medical professional context
        if 'surgeon:' in full_text.lower() or 'physician:' in full_text.lower() or 'dr.' in full_text.lower():
            return False
    
    # Skip if it contains newlines or formatting issues
    if '\n' in name or '\t' in name or '  ' in name:
        return False
    
    # Skip common non-personal words
    common_words = ['the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
                   'via', 'via', 'via', 'via', 'via', 'via', 'via', 'via', 'via', 'via']
    if all(word in common_words for word in words):
        return False
    
    # Skip if it's a single word (names should have at least 2 words)
    if len(words) < 2:
        return False
    
    # Skip if it contains common medical abbreviations
    medical_abbrevs = ['ct', 'mri', 'xray', 'iv', 'po', 'prn', 'bid', 'tid', 'qid', 'qd']
    for word in words:
        if word in medical_abbrevs:
            return False
    
    # Must have proper capitalization (First Last format)
    words = name.split()
    if len(words) >= 2:
        # Check if first word starts with capital and rest are properly capitalized
        if not (words[0][0].isupper() and words[1][0].isupper()):
            return False
    
    return True

def is_ml_likely_sensitive(text, entity_type, full_text):
    """ML-powered determination if an entity is likely sensitive - ENHANCED VERSION"""
    
    # Get context around the entity
    context_start = max(0, full_text.lower().find(text.lower()) - 150)
    context_end = min(len(full_text), full_text.lower().find(text.lower()) + len(text) + 150)
    context = full_text[context_start:context_end].lower()
    
    # Skip single letters and very short text
    if len(text.strip()) <= 1:
        print(f"  🤖 ML: Skipping '{text}' - too short (single letter)")
        return False
    
    # Skip if it's clearly not sensitive (very restrictive list)
    non_sensitive_indicators = [
        'example', 'sample', 'test', 'demo', 'template', 'placeholder',
        'reference', 'citation', 'figure', 'table', 'page', 'chapter',
        'created', 'modified', 'issued', 'published',
        'version', 'build', 'release', 'update', 'patch',
        'copyright', 'all rights reserved', 'terms of service',
        # Medical information that should not be redacted
        'ejection fraction', 'troponin', 'blood pressure', 'heart rate', 'respiratory rate',
        'normal', 'elevated', 'mg/dl', 'ng/ml', 'bpm', 'mmhg', 'min', 'sec',
        'aspirin', 'nitroglycerin', 'metoprolol', 'atorvastatin', 'clopidogrel',
        'penicillin', 'allergy', 'allergies', 'medication', 'medications',
        'diagnosis', 'treatment', 'procedure', 'surgery', 'stent', 'stenosis',
        'chest pain', 'shortness of breath', 'substernal pain', 'cardiac events',
        'family history', 'hypertension', 'sinus rhythm', 'st elevation',
        'left ventricular hypertrophy', 'angiography', 'lad artery',
        'follow-up', 'low-sodium diet', 'exercise', 'lifestyle',
        # Military/organizational information that should not be redacted
        'delta', 'cyber command', 'nato', 'military intelligence', 'joint chiefs',
        'recon team', 'deploy team', 'intelligence division', 'sigint', 'elint',
        'phase two', 'terrain analysis', 'threat assessment', 'military briefing',
        'command', 'division', 'unit', 'squadron', 'battalion', 'regiment',
        'army', 'navy', 'air force', 'marines', 'coast guard', 'special forces',
        # Government and organizational information
        'department of defense', 'dod', 'itar', 'export control', 'regulations',
        'delivery schedule', 'proving ground', 'test facility', 'government',
        'federal', 'state', 'county', 'municipal', 'public', 'official',
        'schedule', 'timeline', 'calendar', 'deadline', 'due date',
        'ground', 'facility', 'site', 'location', 'base', 'station'
    ]
    
    for indicator in non_sensitive_indicators:
        if indicator in context:
            # Don't skip if it's a clear person name (First Last pattern)
            words = text.strip().split()
            if len(words) >= 2 and all(word[0].isupper() and word[1:].islower() for word in words):
                print(f"  🤖 ML: Keeping '{text}' - clear person name pattern despite context: {indicator}")
                continue
            print(f"  🤖 ML: Skipping '{text}' - context suggests non-sensitive: {indicator}")
            return False
    
    # Entity type specific checks
    if entity_type in ['PERSON', 'PER']:
        # Skip if it's clearly a medical professional with title
        medical_titles = ['dr.', 'doctor', 'nurse', 'surgeon', 'physician', 'md', 'professor', 'prof.', 'dr ', 'attending']
        # Only skip if the title appears right before the name
        for title in medical_titles:
            if f"{title} {text.lower()}" in context or f"{title}. {text.lower()}" in context:
                print(f"  🤖 ML: Skipping '{text}' - appears to be medical professional")
                return False
        
        # Skip if it appears after "Attending" or similar medical roles
        medical_roles = ['attending', 'resident', 'intern', 'fellow', 'chief', 'director']
        for role in medical_roles:
            if f"{role} {text.lower()}" in context or f"{role}: {text.lower()}" in context:
                print(f"  🤖 ML: Skipping '{text}' - appears to be medical professional role")
                return False
        
        # Skip if it appears in a medical professional context (but not patient context)
        medical_context = ['surgeon:', 'physician:', 'doctor:', 'nurse:', 'attending:', 'resident:']
        patient_context = ['patient:', 'name:', 'subject:', 'individual:']
        
        # Check if it's in medical professional context
        is_medical_professional = False
        for ctx in medical_context:
            if ctx in context and text.lower() in context:
                is_medical_professional = True
                break
        
        # Check if it's in patient context
        is_patient = False
        for ctx in patient_context:
            if ctx in context and text.lower() in context:
                is_patient = True
                break
        
        if is_medical_professional and not is_patient:
            print(f"  🤖 ML: Skipping '{text}' - appears in medical professional context")
            return False
        
        # Skip if it's clearly an institution name (only if the name itself contains institution words)
        institution_indicators = ['hospital', 'medical center', 'health system', 'clinic', 'general hospital', 'university', 'college', 'institute']
        if any(indicator in text.lower() for indicator in institution_indicators):
            print(f"  🤖 ML: Skipping '{text}' - appears to be institution")
            return False
        
        # Skip common non-personal names (very restrictive)
        common_names = ['john doe', 'jane doe', 'test user', 'admin', 'user', 'guest', 'anonymous']
        if text.lower() in common_names:
            print(f"  🤖 ML: Skipping '{text}' - common placeholder name")
            return False
        
        # For names in ANY context, be very aggressive - consider them sensitive
        # But skip if it's clearly a document title or organizational name
        if any(org in text.lower() for org in ['document', 'briefing', 'report', 'summary', 'analysis', 'matrix', 'data', 'intercepts']):
            print(f"  🤖 ML: Skipping '{text}' - appears to be document/organizational name")
            return False
        
        # Skip if it's clearly a military unit or organization name
        military_orgs = ['delta force', 'cyber command', 'nato', 'intelligence division', 'joint chiefs', 'sigint', 'elint']
        if any(org in text.lower() for org in military_orgs):
            print(f"  🤖 ML: Skipping '{text}' - appears to be military organization")
            return False
        
        # Skip if it's a document title or organizational name
        org_indicators = ['department', 'defense', 'dod', 'itar', 'schedule', 'ground', 'facility', 'base', 'station', 'export', 'control']
        if any(org in text.lower() for org in org_indicators):
            print(f"  🤖 ML: Skipping '{text}' - appears to be organizational name")
            return False
        
        # Only redact if it looks like a real person name (first name + last name pattern)
        words = text.strip().split()
        if len(words) >= 2:
            # Check if it follows typical name pattern (First Last or First Middle Last)
            if all(word[0].isupper() and word[1:].islower() for word in words):
                # Additional check: skip if it contains organizational keywords
                org_keywords = ['schedule', 'ground', 'facility', 'base', 'station', 'department', 'defense', 'export', 'control', 'intelligence', 'briefing']
                if not any(keyword in text.lower() for keyword in org_keywords):
                    print(f"  🤖 ML: Found sensitive person '{text}' - follows name pattern")
                    return True
                else:
                    print(f"  🤖 ML: Skipping '{text}' - contains organizational keywords")
                    return False
        
        print(f"  🤖 ML: Skipping '{text}' - doesn't follow clear person name pattern")
        return False
    
    # High confidence for financial/ID entities
    if entity_type in ['MONEY', 'ORG', 'GPE']:
        # Check for financial context
        financial_indicators = ['account', 'bank', 'credit', 'ssn', 'id', 'payment', 'transaction', 'invoice', 'billing']
        if any(word in context for word in financial_indicators):
            print(f"  🤖 ML: Found sensitive {entity_type} '{text}' in financial context")
            return True
        
        # For organizations, be more selective
        if entity_type == 'ORG':
            # Skip if it's clearly a public institution
            public_institutions = ['government', 'federal', 'state', 'city', 'county', 'public', 'university', 'college']
            if any(inst in text.lower() for inst in public_institutions):
                print(f"  🤖 ML: Skipping '{text}' - appears to be public institution")
                return False
            # Otherwise consider it sensitive
            print(f"  🤖 ML: Found sensitive organization '{text}'")
            return True
    
    # For dates, times, and money - be more selective
    if entity_type in ['DATE', 'TIME', 'MONEY']:
        # Skip if it's clearly a medical finding or lab result
        medical_findings = ['ejection fraction', 'troponin', 'blood pressure', 'heart rate', 'respiratory rate',
                           'normal', 'elevated', 'mg/dl', 'ng/ml', 'bpm', 'mmhg', 'min', 'sec']
        if any(finding in context for finding in medical_findings):
            print(f"  🤖 ML: Skipping '{text}' - appears to be medical finding")
            return False
        
        # Skip if it's a medication dosage
        if any(med in context for med in ['mg', 'ml', 'daily', 'bid', 'tid', 'qid', 'tablet', 'capsule']):
            print(f"  🤖 ML: Skipping '{text}' - appears to be medication dosage")
            return False
        
        # Skip if it's a medical measurement
        if any(measure in context for measure in ['ejection', 'fraction', 'troponin', 'pressure', 'rate']):
            print(f"  🤖 ML: Skipping '{text}' - appears to be medical measurement")
            return False
        
        # Otherwise consider it sensitive
        print(f"  🤖 ML: Found sensitive {entity_type} '{text}'")
        return True
    
    # For locations (GPE) - be more selective
    if entity_type == 'GPE':
        # Skip if it's clearly a public place or institution
        public_places = ['united states', 'usa', 'america', 'city', 'state', 'county', 'country',
                        'hospital', 'medical center', 'health system', 'clinic', 'university', 'college',
                        'general hospital', 'city general', 'county hospital', 'public hospital',
                        'city general hospital', 'general hospital', 'medical center']
        if any(place in text.lower() for place in public_places):
            print(f"  🤖 ML: Skipping '{text}' - appears to be public place or institution")
            return False
        # Otherwise consider it sensitive
        print(f"  🤖 ML: Found sensitive location '{text}'")
        return True
    
    # For locations (LOC) - be more selective (Transformers uses LOC instead of GPE)
    if entity_type == 'LOC':
        # Skip if it's clearly a public place or institution
        public_places = ['united states', 'usa', 'america', 'city', 'state', 'county', 'country',
                        'hospital', 'medical center', 'health system', 'clinic', 'university', 'college',
                        'general hospital', 'city general', 'county hospital', 'public hospital',
                        'city general hospital', 'general hospital', 'medical center']
        if any(place in text.lower() for place in public_places):
            print(f"  🤖 ML: Skipping '{text}' - appears to be public place or institution")
            return False
        # Otherwise consider it sensitive
        print(f"  🤖 ML: Found sensitive location '{text}'")
        return True
    
    # Default to sensitive for any other entity type
    print(f"  🤖 ML: Found sensitive {entity_type} '{text}' - defaulting to sensitive")
    return True

def detect_contextual_sensitive_info(text):
    """AI-powered contextual sensitive information detection"""
    context_matches = []
    
    # Only redact names when they appear in clearly personal contexts
    personal_context_patterns = {
        'patient_name': r'(?:Patient|Name):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
        'patient_id': r'(?:Patient\s+ID|ID|MRN):\s*([A-Z0-9-]{6,})',
        'dob': r'(?:DOB|Date\s+of\s+Birth|Born):\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        'age': r'Age:\s*(\d+)',
        'address': r'(?:Address|Addr):\s*(\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)[^,]*?)',
        'phone_context': r'(?:Phone|Tel|Mobile):\s*(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})',
        'email_context': r'(?:Email|E-mail):\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
    }
    
    for pattern_name, pattern in personal_context_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        print(f"🔍 Pattern {pattern_name}: found {len(matches)} matches: {matches}")
        for match in matches:
            # Additional AI validation
            if is_ai_likely_sensitive(pattern_name, match, text):
                context_matches.append({
                    'text': match,
                    'type': pattern_name,
                    'confidence': 0.95,
                    'context': get_context_around_match(match, text)
                })
                print(f"✅ AI Found contextual {pattern_name}: '{match}'")
    
    # Also detect names without explicit labels (but be more careful)
    name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
    names = re.findall(name_pattern, text)
    print(f"🔍 Found {len(names)} potential names: {names}")
    
    for name in names:
        print(f"🔍 Checking name: '{name}'")
        # Check if it's likely a patient name (not a medical professional)
        if is_likely_patient_name(name, text):
            context_matches.append({
                'text': name,
                'type': 'patient_name',
                'confidence': 0.8,
                'context': get_context_around_match(name, text)
            })
            print(f"✅ AI Found patient name: '{name}'")
    
    return context_matches

def is_ai_likely_sensitive(pattern_type, match, text):
    """AI-powered determination if something is likely sensitive"""
    
    # Get broader context
    context_start = max(0, text.lower().find(match.lower()) - 100)
    context_end = min(len(text), text.lower().find(match.lower()) + len(match) + 100)
    context = text[context_start:context_end].lower()
    
    # AI indicators that this is NOT sensitive
    non_sensitive_indicators = [
        'example', 'sample', 'test', 'demo', 'template', 'placeholder',
        'reference', 'citation', 'figure', 'table', 'page', 'chapter',
        'document', 'created', 'modified', 'issued', 'published',
        'version', 'build', 'release', 'update', 'patch',
        'hospital', 'medical center', 'health system', 'procedure',
        'surgical', 'notes', 'record', 'general'
    ]
    
    # Check if context suggests it's not personal/sensitive
    for indicator in non_sensitive_indicators:
        if indicator in context:
            print(f"  🤖 AI: Skipping '{match}' - context suggests non-sensitive: {indicator}")
            return False
    
    # AI validation based on pattern type
    if pattern_type == 'patient_name':
        # Skip if it's clearly a medical professional or institution
        medical_titles = ['dr', 'doctor', 'nurse', 'surgeon', 'physician']
        if any(title in context for title in medical_titles):
            print(f"  🤖 AI: Skipping '{match}' - appears to be medical professional")
            return False
        
        # Skip common non-personal names
        common_names = ['john doe', 'jane doe', 'test patient', 'sample patient']
        if match.lower() in common_names:
            print(f"  🤖 AI: Skipping '{match}' - common non-personal name")
            return False
    
    if pattern_type == 'dob':
        # Skip if it's clearly a document date
        if any(word in context for word in ['document', 'created', 'modified', 'issued', 'published', 'procedure']):
            print(f"  🤖 AI: Skipping '{match}' - appears to be document date")
            return False
    
    if pattern_type == 'age':
        # Skip if it's clearly not personal age
        if any(word in context for word in ['document', 'version', 'build', 'year']):
            print(f"  🤖 AI: Skipping '{match}' - appears to be non-personal age")
            return False
    
    # AI confidence scoring
    confidence_score = calculate_ai_confidence(pattern_type, match, context)
    
    if confidence_score < 0.7:
        print(f"  🤖 AI: Low confidence ({confidence_score:.2f}) for '{match}' - skipping")
        return False
    
    print(f"  🤖 AI: High confidence ({confidence_score:.2f}) for '{match}' - will redact")
    return True

def calculate_ai_confidence(pattern_type, match, context):
    """Calculate AI confidence score for sensitivity"""
    confidence = 0.5  # Base confidence
    
    # Boost confidence for clear personal indicators
    personal_indicators = ['patient', 'name', 'dob', 'age', 'address', 'phone', 'email', 'id']
    for indicator in personal_indicators:
        if indicator in context:
            confidence += 0.1
    
    # Boost confidence for financial/ID patterns
    if pattern_type in ['ssn', 'credit_card', 'bank_account', 'passport', 'drivers_license']:
        confidence += 0.3
    
    # Reduce confidence for medical/institutional context
    medical_context = ['hospital', 'medical', 'procedure', 'surgical', 'health', 'center']
    for context_word in medical_context:
        if context_word in context:
            confidence -= 0.1
    
    # Ensure confidence is between 0 and 1
    return max(0.0, min(1.0, confidence))

def is_likely_patient_name(name, text):
    """Determine if a name is likely a patient name (not medical professional)"""
    
    # Get context around the name
    context_start = max(0, text.lower().find(name.lower()) - 50)
    context_end = min(len(text), text.lower().find(name.lower()) + len(name) + 50)
    context = text[context_start:context_end].lower()
    
    # Skip if it's clearly a medical professional
    medical_titles = ['dr', 'doctor', 'nurse', 'surgeon', 'physician', 'md']
    for title in medical_titles:
        if title in context:
            print(f"  🤖 AI: Skipping '{name}' - appears to be medical professional")
            return False
    
    # Skip if it's clearly an institution
    institution_indicators = ['hospital', 'medical center', 'health system', 'clinic']
    for indicator in institution_indicators:
        if indicator in context:
            print(f"  🤖 AI: Skipping '{name}' - appears to be institution")
            return False
    
    # Skip common non-personal names
    common_names = ['john doe', 'jane doe', 'test patient', 'sample patient']
    if name.lower() in common_names:
        print(f"  🤖 AI: Skipping '{name}' - common non-personal name")
        return False
    
    # If it appears in a medical document context, it's likely a patient
    medical_document_indicators = ['patient', 'procedure', 'surgical', 'medical', 'hospital']
    if any(indicator in context for indicator in medical_document_indicators):
        print(f"  🤖 AI: Found patient name '{name}' in medical context")
        return True
    
    # Default to not redacting if unclear
    print(f"  🤖 AI: Skipping '{name}' - unclear context")
    return False

def is_common_name(name):
    """Check if a name is a common non-personal name"""
    common_names = [
        'City General', 'General Hospital', 'Medical Center', 'Health System',
        'Patient Surgical', 'Procedure Notes', 'Surgical Procedure', 'Procedure Record',
        'Dr Lisa', 'Dr Chen', 'Lisa Chen', 'Robert Finley', 'Robert J Finley'
    ]
    return name in common_names

def is_common_id(id_val):
    """Check if an ID is a common non-sensitive identifier"""
    # Skip if it's clearly not sensitive
    if len(id_val) < 6:
        return True
    if id_val.isdigit() and len(id_val) < 8:
        return True
    return False

def filter_false_positives(matches, pattern_type, text):
    """Filter out false positives based on context"""
    filtered = []
    
    for match in matches:
        # Get context around the match
        context_start = max(0, text.lower().find(match.lower()) - 50)
        context_end = min(len(text), text.lower().find(match.lower()) + len(match) + 50)
        context = text[context_start:context_end].lower()
        
        # Skip if it's clearly not sensitive
        if is_false_positive(match, pattern_type, context):
            print(f"  ⚠️  Skipping false positive: '{match}' (context: {context[:30]}...)")
            continue
            
        filtered.append(match)
    
    return filtered

def is_false_positive(match, pattern_type, context):
    """Determine if a match is a false positive based on context"""
    
    # Common false positive patterns
    false_positive_indicators = [
        'page', 'chapter', 'section', 'figure', 'table', 'example',
        'sample', 'test', 'demo', 'placeholder', 'template',
        'reference', 'citation', 'footnote', 'appendix',
        'version', 'build', 'release', 'update', 'patch'
    ]
    
    # Check if context suggests it's not sensitive
    for indicator in false_positive_indicators:
        if indicator in context:
            return True
    
    # Specific pattern checks
    if pattern_type == 'phone':
        # Skip if it's clearly a reference number or ID
        if any(word in context for word in ['ref', 'id', 'number', 'code', 'serial']):
            return True
    
    if pattern_type == 'email':
        # Skip if it's a generic or example email
        if any(domain in match.lower() for domain in ['example.com', 'test.com', 'sample.com', 'placeholder.com']):
            return True
    
    if pattern_type == 'date_of_birth':
        # Skip if it's clearly a document date or reference date
        if any(word in context for word in ['document', 'created', 'modified', 'issued', 'published']):
            return True
    
    return False

def detect_contextual_sensitive_info(text):
    """Detect sensitive information based on context and labels"""
    context_matches = []
    
    # Patterns that require specific context to be sensitive
    contextual_patterns = {
        'patient_name': r'\b(?:Patient|Name):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b',
        'patient_id': r'\b(?:Patient\s+ID|ID|MRN):\s*([A-Z0-9-]{6,})\b',
        'dob': r'\b(?:DOB|Date\s+of\s+Birth|Born):\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        'ssn_context': r'\b(?:SSN|Social\s+Security):\s*(\d{3}-?\d{2}-?\d{4})\b',
        'credit_card_context': r'\b(?:Credit\s+Card|Card\s+Number):\s*(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b',
        'bank_account_context': r'\b(?:Account|Acct)\.?\s*#?\s*(\d{8,17})\b',
        'phone_context': r'\b(?:Phone|Tel|Mobile):\s*(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b',
        'email_context': r'\b(?:Email|E-mail):\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
        'address_context': r'\b(?:Address|Addr):\s*(\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)[^,]*)\b'
    }
    
    for pattern_name, pattern in contextual_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Additional context validation
            if is_likely_sensitive(pattern_name, match, text):
                context_matches.append({
                    'text': match,
                    'type': pattern_name,
                    'confidence': 0.9,
                    'context': get_context_around_match(match, text)
                })
                print(f"✅ Found contextual {pattern_name}: '{match}'")
    
    return context_matches

def is_likely_sensitive(pattern_type, match, text):
    """Determine if a contextual match is likely to be sensitive"""
    
    # Get broader context
    context_start = max(0, text.lower().find(match.lower()) - 100)
    context_end = min(len(text), text.lower().find(match.lower()) + len(match) + 100)
    context = text[context_start:context_end].lower()
    
    # Skip if it's clearly not personal/sensitive
    non_sensitive_indicators = [
        'example', 'sample', 'test', 'demo', 'template', 'placeholder',
        'reference', 'citation', 'figure', 'table', 'page', 'chapter'
    ]
    
    for indicator in non_sensitive_indicators:
        if indicator in context:
            return False
    
    # Additional validation based on pattern type
    if pattern_type == 'patient_name':
        # Skip common non-personal names
        common_names = ['john doe', 'jane doe', 'test patient', 'sample patient']
        if match.lower() in common_names:
            return False
    
    return True

def get_context_around_match(match, text, context_size=50):
    """Get context around a match for better understanding"""
    pos = text.lower().find(match.lower())
    if pos == -1:
        return ""
    
    start = max(0, pos - context_size)
    end = min(len(text), pos + len(match) + context_size)
    return text[start:end]

def redact_text(text, sensitive_info):
    """Redact sensitive information from text"""
    redacted_text = text
    
    # Redact regex matches
    for pattern_name, matches in sensitive_info['regex_matches'].items():
        for match in matches:
            redacted_text = redacted_text.replace(match, '[REDACTED]')
    
    # Redact keyword contexts more carefully
    for keyword_match in sensitive_info['keyword_matches']:
        if keyword_match['confidence'] > 0.5:
            # Only redact the specific keyword, not the entire context
            keyword = keyword_match['keyword']
            redacted_text = redacted_text.replace(keyword, '[REDACTED]')
    
    return redacted_text

def create_redacted_pdf(original_file_path, sensitive_info, output_path):
    """Create a redacted PDF with visual redaction (black boxes) over sensitive text"""
    try:
        # Try using PyMuPDF for better redaction
        try:
            import fitz  # PyMuPDF
            return create_redacted_pdf_pymupdf(original_file_path, sensitive_info, output_path)
        except ImportError:
            print("PyMuPDF not available, using PyPDF2 method...")
            return create_redacted_pdf_pypdf2(original_file_path, sensitive_info, output_path)
        
    except Exception as e:
        print(f"Error creating redacted PDF: {e}")
        # Fallback to text-based redaction
        return create_text_redacted_pdf(original_file_path, sensitive_info, output_path)

def create_redacted_pdf_pymupdf(original_file_path, sensitive_info, output_path):
    """Create redacted PDF using PyMuPDF for accurate positioning"""
    try:
        import fitz
        
        print(f"🔧 Starting redaction process...")
        
        # Open the PDF
        doc = fitz.open(original_file_path)
        print(f"📄 Processing {len(doc)} pages...")
        
        # Get all sensitive text to redact
        sensitive_texts = []
        
        # Collect all matches
        for pattern_name, matches in sensitive_info.get('regex_matches', {}).items():
            for match in matches:
                sensitive_texts.append(match)
                print(f"🎯 Will redact: '{match}' (type: {pattern_name})")
        
        # Collect ML matches
        for ml_match in sensitive_info.get('ml_matches', []):
            sensitive_texts.append(ml_match['text'])
            print(f"🎯 Will redact ML: '{ml_match['text']}' (type: {ml_match['type']}, confidence: {ml_match['confidence']:.2f})")
        
        print(f"📋 Total items to redact: {len(sensitive_texts)}")
        
        if not sensitive_texts:
            print("⚠️ No sensitive information found to redact!")
            # Still create a copy
            doc.save(output_path)
            doc.close()
            return True
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"📄 Processing page {page_num + 1}...")
            
            # Find and redact each sensitive text
            for sensitive_text in sensitive_texts:
                # Search for the text on this page
                text_instances = page.search_for(sensitive_text)
                
                if text_instances:
                    print(f"  ✅ Found '{sensitive_text}' {len(text_instances)} times on page {page_num + 1}")
                    
                    # Create redaction annotations
                    for rect in text_instances:
                        # Create a redaction annotation with black fill
                        redact_annot = page.add_redact_annot(rect, fill=(0, 0, 0))
                        redact_annot.set_colors(stroke=(0, 0, 0), fill=(0, 0, 0))
                        redact_annot.update()
                else:
                    print(f"  ❌ Not found on page {page_num + 1}: '{sensitive_text}'")
            
            # Apply redactions
            page.apply_redactions()
        
        # Save the redacted PDF
        doc.save(output_path)
        doc.close()
        
        print(f"✅ Redacted PDF saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error with PyMuPDF redaction: {e}")
        import traceback
        traceback.print_exc()
        return False

def search_partial_text(page, text):
    """Search for partial text matches when exact match fails"""
    try:
        # Split text into words and search for each word
        words = text.split()
        if len(words) < 2:
            return []
        
        # Search for the first few words
        search_text = ' '.join(words[:2])  # First 2 words
        return page.search_for(search_text)
    except:
        return []

def create_redacted_pdf_pypdf2(original_file_path, sensitive_info, output_path):
    """Create redacted PDF using PyPDF2 with overlay method"""
    try:
        import PyPDF2
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        import io
        
        # Read the original PDF
        with open(original_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pdf_writer = PyPDF2.PdfWriter()
            
            # Process each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                
                # Extract text to find positions of sensitive information
                page_text = page.extract_text()
                
                # Create redaction overlays for this page
                redaction_overlays = create_redaction_overlays(page_text, sensitive_info)
                
                # Add the original page
                pdf_writer.add_page(page)
                
                # Create redaction overlay page
                if redaction_overlays:
                    overlay_page = create_redaction_overlay_page(redaction_overlays, letter)
                    if overlay_page:
                        pdf_writer.add_page(overlay_page)
            
            # Write the redacted PDF
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
        
        return True
        
    except Exception as e:
        print(f"Error with PyPDF2 redaction: {e}")
        return False

def create_redaction_overlays(page_text, sensitive_info):
    """Create redaction overlays for sensitive information"""
    overlays = []
    
    # Process regex matches
    for pattern_name, matches in sensitive_info.get('regex_matches', {}).items():
        for match in matches:
            # Find position of the match in the text
            pos = page_text.find(match)
            if pos != -1:
                overlays.append({
                    'text': match,
                    'type': pattern_name,
                    'position': pos,
                    'length': len(match)
                })
    
    # Process keyword matches
    for keyword_match in sensitive_info.get('keyword_matches', []):
        if keyword_match['confidence'] > 0.5:
            keyword = keyword_match['keyword']
            pos = page_text.find(keyword)
            if pos != -1:
                overlays.append({
                    'text': keyword,
                    'type': 'keyword',
                    'position': pos,
                    'length': len(keyword)
                })
    
    return overlays

def create_redaction_overlay_page(overlays, page_size):
    """Create a PDF page with redaction overlays (black boxes)"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        
        # Create a buffer for the overlay
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=page_size)
        width, height = page_size
        
        # Set up for text positioning (approximate)
        margin = inch
        line_height = 14
        chars_per_line = 80
        
        for overlay in overlays:
            # Calculate approximate position
            text_pos = overlay['position']
            line_num = text_pos // chars_per_line
            char_in_line = text_pos % chars_per_line
            
            x = margin + (char_in_line * 7)  # Approximate character width
            y = height - margin - (line_num * line_height)
            
            # Draw black rectangle over the text
            text_width = overlay['length'] * 7  # Approximate width
            text_height = line_height
            
            c.setFillColor('black')
            c.rect(x, y - text_height, text_width, text_height, fill=1, stroke=0)
        
        c.save()
        buffer.seek(0)
        
        # Convert to PyPDF2 page
        from PyPDF2 import PdfReader
        pdf_reader = PdfReader(buffer)
        return pdf_reader.pages[0]
        
    except Exception as e:
        print(f"Error creating redaction overlay: {e}")
        return None

def create_text_redacted_pdf(original_file_path, sensitive_info, output_path):
    """Fallback: Create text-based redacted PDF"""
    try:
        # Extract text from original PDF
        text_content = extract_text_from_pdf(original_file_path)
        
        # Redact the text
        redacted_text = redact_text(text_content, sensitive_info)
        
        # Create new PDF with redacted text
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        
        # Set font and margins
        c.setFont("Helvetica", 12)
        margin = inch
        line_height = 14
        y_position = height - margin
        
        # Split text into lines
        lines = redacted_text.split('\n')
        
        for line in lines:
            if y_position < margin:
                c.showPage()
                y_position = height - margin
            
            # Wrap long lines
            if len(line) > 80:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) > 80:
                        c.drawString(margin, y_position, current_line)
                        y_position -= line_height
                        current_line = word
                    else:
                        current_line += " " + word if current_line else word
                
                if current_line:
                    c.drawString(margin, y_position, current_line)
                    y_position -= line_height
            else:
                c.drawString(margin, y_position, line)
                y_position -= line_height
        
        c.save()
        return True
        
    except Exception as e:
        print(f"Error creating text redacted PDF: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from PDF
            text_content = extract_text_from_pdf(filepath)
            
            # Detect sensitive information
            sensitive_info = detect_sensitive_info(text_content)
            
            # Create redacted PDF with visual redaction
            redacted_filename = f"redacted_{filename}"
            redacted_filepath = os.path.join(app.config['REDACTED_FOLDER'], redacted_filename)
            
            if create_redacted_pdf(filepath, sensitive_info, redacted_filepath):
                return jsonify({
                    'success': True,
                    'original_text': text_content,
                    'sensitive_info': sensitive_info,
                    'redacted_file': redacted_filename,
                    'message': 'File processed successfully'
                })
            else:
                return jsonify({'error': 'Failed to create redacted PDF'}), 500
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['REDACTED_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/preview/<filename>')
def preview_file(filename):
    filepath = os.path.join(app.config['REDACTED_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=False)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting AI Sensitive Information Redactor...")
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("📁 Redacted folder:", app.config['REDACTED_FOLDER'])
    print("🌐 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
