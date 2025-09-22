# Synthetic Data Generation Approaches for Contract Classification

## 🎯 Overview
Generate synthetic training data to address data scarcity in contract correspondence classification while preventing overfitting.

## 🔧 Available Approaches

### 1. **LLM-Based Generation** ⭐ Recommended
**Libraries:** `openai`, `anthropic`, `transformers`

```python
# Using OpenAI GPT-4
from openai import OpenAI
client = OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{
        "role": "user", 
        "content": "Generate 5 contract correspondence samples for 'Payment Delay' issues..."
    }],
    temperature=0.8,  # Higher for variety
    response_format={"type": "json_object"}
)
```

**Pros:**
- High quality, contextually relevant data
- Natural language variation
- Domain knowledge incorporation
- Scalable generation

**Cons:**
- API costs
- Rate limiting
- Requires good prompt engineering

### 2. **Template-Based Variation**
**Libraries:** `jinja2`, `faker`, `random`

```python
from faker import Faker
import random

fake = Faker()

template = """
Subject: {issue_type} - Project {project_id}
Dear {authority},
We are writing regarding {issue_description} for {project_name}.
Amount involved: Rs. {amount:,}
Completion date: {date}
"""

# Generate variations
for i in range(10):
    sample = template.format(
        issue_type=random.choice(['Payment Delay', 'Scope Change']),
        project_id=fake.random_int(1000, 9999),
        authority=fake.name(),
        issue_description=fake.text(max_nb_chars=200),
        project_name=fake.company(),
        amount=fake.random_int(100000, 10000000),
        date=fake.date()
    )
```

### 3. **Data Augmentation Libraries**
**Libraries:** `nlpaug`, `textattack`, `transformers`

```python
# Text augmentation using nlpaug
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

# Synonym replacement
syn_aug = naw.SynonymAug(aug_src='wordnet')
augmented = syn_aug.augment(original_text)

# Back translation for paraphrasing
back_trans_aug = nas.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de',
    to_model_name='facebook/wmt19-de-en'
)
paraphrased = back_trans_aug.augment(original_text)

# BERT-based word replacement
bert_aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action="substitute"
)
contextual_aug = bert_aug.augment(original_text)
```

### 4. **Web Scraping Approaches**
**Libraries:** `scrapy`, `beautifulsoup4`, `requests`, `selenium`

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Scrape contract templates and legal documents
def scrape_contract_samples():
    urls = [
        "https://example-legal-docs.com",
        "https://construction-contracts.com",
        # Add relevant legal/contract websites
    ]
    
    samples = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract relevant text sections
        for section in soup.find_all('div', class_='contract-text'):
            samples.append({
                'text': section.get_text(),
                'source': url
            })
    
    return samples
```

### 5. **Generative AI Libraries**
**Libraries:** `transformers`, `diffusers`, `sentence-transformers`

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Fine-tuned text generation
generator = pipeline('text-generation', 
                    model='gpt2-medium',
                    tokenizer='gpt2-medium')

# Generate contract-style text
prompt = "Contract correspondence regarding payment delay:"
generated = generator(prompt, 
                     max_length=200, 
                     num_return_sequences=5,
                     temperature=0.8)

# Domain-specific model fine-tuning
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
```

### 6. **Specialized Synthetic Data Libraries**
**Libraries:** `synthcity`, `sdv`, `gretel-synthetics`

```python
# Using SDV (Synthetic Data Vault)
from sdv.tabular import GaussianCopula
from sdv.constraints import FixedCombinations

# Create synthetic tabular data
model = GaussianCopula()
model.fit(training_data)
synthetic_data = model.sample(num_rows=1000)

# Using Gretel for text
from gretel_synthetics.timeseries import config_from_model_config
from gretel_synthetics.batch import DataFrameBatch

config = config_from_model_config(base_config='synthetics/tabular-lstm')
batch = DataFrameBatch(df=training_df, config=config)
batch.create_job()
synthetic_df = batch.get_synthetic_data()
```

## 🚫 Overfitting Prevention Strategies

### 1. **Cross-Validation with Holdout Sets**
```python
from sklearn.model_selection import StratifiedKFold

# Ensure real data in validation
def create_robust_splits(real_data, synthetic_data):
    # Always keep some real data for validation
    real_train, real_val = train_test_split(real_data, test_size=0.3)
    
    # Add synthetic only to training
    train_combined = pd.concat([real_train, synthetic_data])
    
    return train_combined, real_val  # Validation is pure real data
```

### 2. **Synthetic Data Ratio Control**
```python
def limit_synthetic_ratio(real_samples, synthetic_samples, max_ratio=2.0):
    """Limit synthetic data to avoid overwhelming real patterns"""
    max_synthetic = int(len(real_samples) * max_ratio)
    if len(synthetic_samples) > max_synthetic:
        return synthetic_samples.sample(n=max_synthetic, random_state=42)
    return synthetic_samples
```

### 3. **Diversity Enforcement**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def ensure_diversity(synthetic_samples, min_similarity=0.3):
    """Remove too-similar synthetic samples"""
    vectorizer = TfidfVectorizer(max_features=1000)
    vectors = vectorizer.fit_transform(synthetic_samples['text'])
    
    # Remove samples that are too similar
    keep_indices = []
    for i, vector in enumerate(vectors):
        similarities = cosine_similarity(vector, vectors).flatten()
        # Keep if not too similar to previous kept samples
        if not any(similarities[j] > min_similarity for j in keep_indices):
            keep_indices.append(i)
    
    return synthetic_samples.iloc[keep_indices]
```

### 4. **Progressive Training**
```python
def progressive_synthetic_training(model, real_data, synthetic_data_batches):
    """Gradually introduce synthetic data during training"""
    
    # Start with real data only
    model.fit(real_data)
    
    # Gradually add synthetic data
    for batch in synthetic_data_batches:
        combined_data = pd.concat([real_data, batch])
        model.partial_fit(combined_data)
        
        # Validate on real data only
        validation_score = model.score(real_validation_data)
        if validation_score < previous_score:
            break  # Stop if performance degrades
    
    return model
```

## 📊 Quality Metrics for Synthetic Data

### 1. **Statistical Similarity**
```python
from scipy import stats

def statistical_comparison(real_data, synthetic_data):
    """Compare distributions statistically"""
    
    # KS test for continuous variables
    ks_stat, p_value = stats.ks_2samp(real_data['length'], 
                                      synthetic_data['length'])
    
    # Chi-square test for categorical variables
    chi2, p_chi2 = stats.chi2_contingency([
        real_data['category'].value_counts(),
        synthetic_data['category'].value_counts()
    ])
    
    return {
        'ks_test': (ks_stat, p_value),
        'chi2_test': (chi2, p_chi2)
    }
```

### 2. **Semantic Similarity**
```python
from sentence_transformers import SentenceTransformer

def semantic_quality_check(real_samples, synthetic_samples):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    real_embeddings = model.encode(real_samples)
    synthetic_embeddings = model.encode(synthetic_samples)
    
    # Calculate centroid distances
    real_centroid = np.mean(real_embeddings, axis=0)
    synthetic_centroid = np.mean(synthetic_embeddings, axis=0)
    
    centroid_distance = cosine_similarity([real_centroid], [synthetic_centroid])[0][0]
    
    return centroid_distance
```

## 🎯 Recommended Implementation Strategy

### Phase 1: Quick Start (Template-Based)
1. Use template-based generation for immediate results
2. Entity substitution for variation
3. Rule-based paraphrasing

### Phase 2: LLM Enhancement (High Quality)
1. Use GPT-4 for critical issue types
2. Domain-specific prompting
3. JSON-structured output

### Phase 3: Advanced Techniques (Scale)
1. Fine-tune domain-specific models
2. Web scraping for domain data
3. Multi-modal generation

### Phase 4: Production Pipeline
1. Automated quality checks
2. Continuous validation
3. A/B testing with real data

## 🔧 Implementation Order

1. **Start with our implemented solution** (`synthetic_data_generator.py`)
2. **Add web scraping** for domain-specific samples
3. **Implement nlpaug** for text augmentation
4. **Fine-tune models** if budget allows
5. **Set up quality monitoring** pipeline

## 💡 Best Practices

1. **Always validate on real data only**
2. **Limit synthetic data ratio (max 2:1)**
3. **Ensure diversity in generation**
4. **Monitor for distribution shift**
5. **Use multiple generation methods**
6. **Regular quality audits**

## 📚 Required Libraries

```bash
# Core generation
pip install openai anthropic transformers

# Augmentation
pip install nlpaug textattack

# Web scraping
pip install scrapy beautifulsoup4 requests selenium

# Synthetic data
pip install sdv synthcity gretel-synthetics

# Quality metrics
pip install sentence-transformers scikit-learn scipy

# Utilities
pip install faker jinja2 pandas numpy
```

This comprehensive approach ensures high-quality synthetic data while preventing overfitting through multiple validation strategies.