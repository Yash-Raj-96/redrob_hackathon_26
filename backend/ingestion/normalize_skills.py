"""
Normalize and standardize skills across candidates
"""
import pandas as pd
from typing import List, Dict, Set
from collections import Counter
import re
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class SkillNormalizer:
    """Normalize skill names and create skill taxonomy"""
    
    def __init__(self):
        self.skill_mapping = {
            # Programming Languages
            'python': 'Python',
            'py': 'Python',
            'python3': 'Python',
            'javascript': 'JavaScript',
            'js': 'JavaScript',
            'java': 'Java',
            'c++': 'C++',
            'cpp': 'C++',
            'csharp': 'C#',
            'c#': 'C#',
            'golang': 'Go',
            'go': 'Go',
            'rust': 'Rust',
            'ruby': 'Ruby',
            'php': 'PHP',
            'swift': 'Swift',
            'kotlin': 'Kotlin',
            'typescript': 'TypeScript',
            'ts': 'TypeScript',
            'scala': 'Scala',
            'r': 'R',
            'matlab': 'MATLAB',
            
            # Data Science & ML
            'machine learning': 'Machine Learning',
            'ml': 'Machine Learning',
            'deep learning': 'Deep Learning',
            'dl': 'Deep Learning',
            'artificial intelligence': 'Artificial Intelligence',
            'ai': 'Artificial Intelligence',
            'data science': 'Data Science',
            'data analytics': 'Data Analytics',
            'nlp': 'Natural Language Processing',
            'natural language processing': 'Natural Language Processing',
            'computer vision': 'Computer Vision',
            'cv': 'Computer Vision',
            'reinforcement learning': 'Reinforcement Learning',
            
            # ML Frameworks
            'tensorflow': 'TensorFlow',
            'tf': 'TensorFlow',
            'pytorch': 'PyTorch',
            'keras': 'Keras',
            'scikit-learn': 'Scikit-learn',
            'sklearn': 'Scikit-learn',
            'huggingface': 'Hugging Face',
            'transformers': 'Transformers',
            'langchain': 'LangChain',
            'llama': 'Llama',
            'openai': 'OpenAI',
            'anthropic': 'Anthropic',
            
            # Big Data
            'spark': 'Apache Spark',
            'apache spark': 'Apache Spark',
            'hadoop': 'Hadoop',
            'hive': 'Hive',
            'kafka': 'Kafka',
            'airflow': 'Airflow',
            'dbt': 'dbt',
            'databricks': 'Databricks',
            'snowflake': 'Snowflake',
            
            # Databases
            'sql': 'SQL',
            'postgresql': 'PostgreSQL',
            'postgres': 'PostgreSQL',
            'mysql': 'MySQL',
            'mongodb': 'MongoDB',
            'mongo': 'MongoDB',
            'redis': 'Redis',
            'elasticsearch': 'Elasticsearch',
            'elastic': 'Elasticsearch',
            'cassandra': 'Cassandra',
            'dynamodb': 'DynamoDB',
            
            # Cloud Platforms
            'aws': 'AWS',
            'amazon web services': 'AWS',
            'azure': 'Azure',
            'microsoft azure': 'Azure',
            'gcp': 'GCP',
            'google cloud': 'GCP',
            'google cloud platform': 'GCP',
            'kubernetes': 'Kubernetes',
            'k8s': 'Kubernetes',
            'docker': 'Docker',
            'terraform': 'Terraform',
            'jenkins': 'Jenkins',
            'gitlab': 'GitLab',
            'github actions': 'GitHub Actions',
            
            # Web Development
            'react': 'React',
            'reactjs': 'React',
            'vue': 'Vue.js',
            'vuejs': 'Vue.js',
            'angular': 'Angular',
            'node': 'Node.js',
            'nodejs': 'Node.js',
            'express': 'Express.js',
            'django': 'Django',
            'flask': 'Flask',
            'fastapi': 'FastAPI',
            'spring': 'Spring Boot',
            'spring boot': 'Spring Boot',
            
            # Soft Skills
            'communication': 'Communication',
            'teamwork': 'Teamwork',
            'leadership': 'Leadership',
            'problem solving': 'Problem Solving',
            'critical thinking': 'Critical Thinking',
            'project management': 'Project Management',
            'agile': 'Agile',
            'scrum': 'Scrum'
        }
        
        self.skill_categories = {
            'Programming Languages': ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'TypeScript', 'Scala', 'R'],
            'Data Science & ML': ['Machine Learning', 'Deep Learning', 'Artificial Intelligence', 'Data Science', 'Data Analytics', 'NLP', 'Computer Vision'],
            'ML Frameworks': ['TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Hugging Face', 'Transformers', 'LangChain'],
            'Big Data': ['Apache Spark', 'Hadoop', 'Kafka', 'Airflow', 'dbt', 'Databricks', 'Snowflake'],
            'Databases': ['SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Cassandra', 'DynamoDB'],
            'Cloud': ['AWS', 'Azure', 'GCP', 'Kubernetes', 'Docker', 'Terraform', 'Jenkins'],
            'Web Development': ['React', 'Vue.js', 'Angular', 'Node.js', 'Express.js', 'Django', 'Flask', 'FastAPI', 'Spring Boot'],
            'Soft Skills': ['Communication', 'Teamwork', 'Leadership', 'Problem Solving', 'Project Management', 'Agile', 'Scrum']
        }
    
    async def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize skills in the dataframe"""
        logger.info("Normalizing skills...")
        
        if 'skills' not in df.columns:
            logger.warning("No skills column found")
            return df
        
        # Apply normalization to each candidate's skills
        df['skills_normalized'] = df['skills'].apply(self._normalize_skill_list)
        df['skills_categorized'] = df['skills_normalized'].apply(self._categorize_skills)
        
        # Create skill count features
        df['num_skills'] = df['skills_normalized'].apply(len)
        df['unique_skill_categories'] = df['skills_categorized'].apply(len)
        
        # Extract top skills across dataset for reference
        all_skills = []
        for skills in df['skills_normalized']:
            all_skills.extend(skills)
        
        skill_counts = Counter(all_skills)
        self.top_skills = dict(skill_counts.most_common(50))
        
        logger.info(f"Normalized {len(df)} candidate profiles with {len(self.skill_mapping)} skill mappings")
        return df
    
    def _normalize_skill_list(self, skills: List[str]) -> List[str]:
        """Normalize a list of skills"""
       # if not skills or pd.isna(skills):
        #    return []
        
        normalized = set()
        for skill in skills:
            if not skill:
                continue
            
            skill_lower = skill.lower().strip()
            
            # Check for direct mapping
            if skill_lower in self.skill_mapping:
                normalized.add(self.skill_mapping[skill_lower])
            else:
                # Try partial matching
                matched = False
                for key, value in self.skill_mapping.items():
                    if key in skill_lower or skill_lower in key:
                        normalized.add(value)
                        matched = True
                        break
                
                if not matched:
                    # Capitalize properly
                    normalized.add(skill.title())
        
        return sorted(list(normalized))
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills into domains"""
        categorized = {}
        
        for skill in skills:
            for category, category_skills in self.skill_categories.items():
                if skill in category_skills:
                    if category not in categorized:
                        categorized[category] = []
                    categorized[category].append(skill)
                    break
            else:
                if 'Other' not in categorized:
                    categorized['Other'] = []
                categorized['Other'].append(skill)
        
        return categorized
    
    def get_skill_taxonomy(self) -> Dict:
        """Return the skill taxonomy"""
        return {
            'mapping': self.skill_mapping,
            'categories': self.skill_categories,
            'top_skills': getattr(self, 'top_skills', {})
        }