import nltk
import ssl

# Bypass SSL verification (temporary workaround)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download stopwords
nltk.download('stopwords')