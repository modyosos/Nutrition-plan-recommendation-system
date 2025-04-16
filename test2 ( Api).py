from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import random
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from collections import defaultdict
import warnings
import uuid
import uvicorn
from typing import Dict, Any, List

app = FastAPI(title="Fat2Fit Meal Planner API")

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

used_meal_ids: set = set()
user_ratings: Dict[str, int] = {}
all_rated_meals: set = set()
low_rated_meals: set = set()
tfidf_vectorizer_global = TfidfVectorizer(stop_words='english')
user_dynamic_weights: Dict[Any, Dict[str, Any]] = defaultdict(lambda: {
    'tfidf_weight': 0.7,
    'macro_weight': 0.3,
    'macro_weights': {
        'protein': 0.4,
        'carbs': 0.3,
        'fat': 0.3
    }
})
meal_lists_global: Dict[str, list] = {}
sessions: Dict[str, Any] = {}

class SessionState:
    def __init__(self):
        self.language: str = 'english'
        self.food_type: str = 'egyptian'
        self.weight: float | None = None
        self.height: float | None = None
        self.age: int | None = None
        self.gender: str | None = None
        self.activity_level: str | None = None
        self.bmr: float | None = None
        self.tdee: float | None = None
        self.user_preferences: list | None = None
        self.calorie_target: int | None = None
        self.weekly_plan: list | None = None
        self.plan_count: int = 0
        self.last_plan_rated_positively: bool = False

class UserInfoInput(BaseModel):
    language: str = Field(..., description="User interface language ('english' or 'arabic')")
    food_type: str = Field(..., description="Preferred food type ('egyptian', 'american', or 'mixed')")
    weight: float = Field(..., gt=0, description="Weight in kilograms")
    height: float = Field(..., gt=0, description="Height in centimeters")
    age: int = Field(..., gt=0, lt=120, description="Age in years")
    gender: str = Field(..., description="Gender ('male' or 'female')")
    activity_level: str = Field(..., description="User's activity level (e.g., 'sedentary', 'lightly_active')")
    body_shape_goal: str = Field(..., description="Desired body shape goal (e.g., 'lean', 'muscular')")

class UserInfoOutput(BaseModel):
    session_id: str
    bmr: float
    tdee: float
    calorie_target: int
    bmi_status: str

class Rating(BaseModel):
    meal_id: str
    rating: int = Field(..., ge=1, le=5)

class FeedbackInput(BaseModel):
    satisfied: bool

class MealNutritionOutput(BaseModel):
    protein: float = Field(..., description="Protein in grams")
    carbs: float = Field(..., description="Carbohydrates in grams")
    fat: float = Field(..., description="Fat in grams")

class MealOutput(BaseModel):
    id: str
    name: str
    calories: int
    nutrition: MealNutritionOutput
    serving_size: str | None = None
    bread_suggestion: str | None = None
    recipe: str | None = None

class DayPlanOutput(BaseModel):
    day: str
    breakfast: MealOutput
    snack: MealOutput
    lunch: MealOutput

class WeeklyPlanOutput(BaseModel):
    weekly_plan: list[DayPlanOutput]

ACTIVITY_LEVELS = {
    'sedentary': {'en': 'Little/no exercise', 'ar': 'قليل/بدون نشاط'},
    'lightly_active': {'en': 'Light exercise 1-3 days/week', 'ar': 'نشاط خفيف 1-3 أيام/أسبوع'},
    'moderately_active': {'en': 'Moderate exercise 3-5 days/week', 'ar': 'نشاط معتدل 3-5 أيام/أسبوع'},
    'very_active': {'en': 'Hard exercise 6-7 days/week', 'ar': 'نشاط قوي 6-7 أيام/أسبوع'},
    'extra_active': {'en': 'Very hard exercise + physical job', 'ar': 'نشاط قوي جدًا + عمل بدني'}
}

BODY_SHAPES = {
    'muscular': {
        'en': "Bulky muscle gain (High protein, high carbs, moderate fat)", 'ar': "اكتساب كتلة عضلية كبيرة (بروتين عالي, كاربوهيدرات عالية, دهون معتدلة)",
        'calorie_factor': 1.15, 'nutrition_prefs': {'body_shape': 'muscular', 'protein': 'high', 'carbs': 'high', 'fat': 'moderate'}
    },
    'lean': {
        'en': "Toned lean muscle (High protein, low carbs, moderate fat)", 'ar': "عضلات متناسقة قليلة الدهون (بروتين عالي, كاربوهيدرات قليلة, دهون معتدلة)",
        'calorie_factor': 0.9, 'nutrition_prefs': {'body_shape': 'lean', 'protein': 'high', 'carbs': 'low', 'fat': 'moderate'}
    },
    'athletic': {
        'en': "Athletic performance (Balanced macros, high protein)", 'ar': "أداء رياضي (توازن في العناصر, بروتين عالي)",
        'calorie_factor': 1.05, 'nutrition_prefs': {'body_shape': 'athletic', 'protein': 'high', 'carbs': 'moderate', 'fat': 'moderate'}
    },
    'weight_loss': {
        'en': "Fat loss (High protein, low carbs & fat)", 'ar': "فقدان دهون (بروتين عالي, كاربوهيدرات ودهون قليلة)",
        'calorie_factor': 0.8, 'nutrition_prefs': {'body_shape': 'weight_loss', 'protein': 'high', 'carbs': 'low', 'fat': 'low'}
    },
    'maintain': {
        'en': "Maintain current weight (Balanced nutrition)", 'ar': "الحفاظ على الوزن الحالي (توازن غذائي)",
        'calorie_factor': 1.0, 'nutrition_prefs': {'body_shape': 'maintain', 'protein': 'moderate', 'carbs': 'moderate', 'fat': 'moderate'}
    }
}

def calculate_bmr(weight, height, age, gender):
    gender_lower = gender.lower()
    if gender_lower in ['male', 'ذكر']:
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    elif gender_lower in ['female', 'أنثى']:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    else:
        raise ValueError("Invalid gender provided for BMR calculation.")

def calculate_tdee(bmr, activity_level):
    activity_factors = {'sedentary': 1.2, 'lightly_active': 1.375, 'moderately_active': 1.55, 'very_active': 1.725, 'extra_active': 1.9}
    return bmr * activity_factors.get(activity_level.lower(), 1.2)

def calculate_nutrition_score(meal, nutrition_prefs, calorie_target, meal_type):
    body_shape = nutrition_prefs.get('body_shape', 'lean')
    if 'nutrition' not in meal or not isinstance(meal['nutrition'], dict): return 0
    protein = meal['nutrition'].get('protein', 0); carbs = meal['nutrition'].get('carbs', 0); fat = meal['nutrition'].get('fat', 0)
    calories = meal.get('calories', 0)
    targets_map = {
        'muscular': {'protein': 30, 'carbs': 50, 'fat': (15,25), 'cal_range': (400,600)}, 'lean': {'protein': 25, 'carbs': 20, 'fat': (10,20), 'cal_range': (300,450)},
        'weight_loss': {'protein': 25, 'carbs': 15, 'fat': (5,15), 'cal_range': (250,400)}, 'athletic': {'protein': 28, 'carbs': 40, 'fat': (12,22), 'cal_range': (350,500)},
        'maintain': {'protein': 20, 'carbs': 35, 'fat': (15,25), 'cal_range': (300,500)}
    }
    targets = targets_map.get(body_shape, targets_map['lean'])
    weights_map = {
        'muscular': {'protein': 0.5, 'carbs': 0.4, 'fat': 0.1}, 'lean': {'protein': 0.6, 'carbs': 0.2, 'fat': 0.2},
        'weight_loss': {'protein': 0.5, 'carbs': 0.3, 'fat': 0.2}, 'athletic': {'protein': 0.5, 'carbs': 0.3, 'fat': 0.2},
        'maintain': {'protein': 0.4, 'carbs': 0.4, 'fat': 0.2}
    }
    weights = weights_map.get(body_shape, weights_map['lean'])
    protein_target = targets.get('protein', 1); protein_score = min((protein / protein_target) * 100, 150) if protein_target > 0 else 0
    carbs_target = targets.get('carbs', 1)
    if body_shape in ['lean', 'weight_loss']: carbs_score = max(0, 100 - ((carbs - carbs_target) * 3)) if carbs_target > 0 else 0
    else: carbs_score = min((carbs / carbs_target) * 100, 100) if carbs_target > 0 else 0
    fat_range = targets.get('fat', (0, 0)); fat_score = 100 if fat_range[0] <= fat <= fat_range[1] else 0
    meal_type_cal_weights = {'breakfast': 0.3, 'lunch': 0.4, 'snack': 0.15}
    cal_target_meal = calorie_target * meal_type_cal_weights.get(meal_type, 0.3)
    calorie_score = max(0, 100 - (abs(calories - cal_target_meal)/cal_target_meal * 100)) if cal_target_meal > 0 else 0
    tag_bonus = 0
    preferred_tags_map = {
        'muscular': ['high-protein', 'high-carb', 'bulking'], 'lean': ['high-protein', 'low-carb', 'keto'],
        'weight_loss': ['low-calorie', 'portion-controlled', 'high-protein'], 'athletic': ['high-protein', 'balanced-macros'], 'maintain': ['balanced-macros']
    }
    preferred_tags = preferred_tags_map.get(body_shape, [])
    meal_tags = meal.get('tags', [])
    if isinstance(meal_tags, list):
        for tag in preferred_tags:
            if tag in meal_tags: tag_bonus += 20
    w_protein = weights.get('protein', 0); w_carbs = weights.get('carbs', 0); w_fat = weights.get('fat', 0)
    final_score = (protein_score * w_protein + carbs_score * w_carbs + fat_score * w_fat + calorie_score * 0.15 + tag_bonus)
    return final_score

def filter_meals_by_body_shape(meals, body_shape, meal_type):
    criteria = {
        'muscular': {'breakfast': lambda m: m.get('nutrition', {}).get('protein', 0) >= 25, 'lunch': lambda m: m.get('nutrition', {}).get('protein', 0) >= 30, 'snack': lambda m: m.get('nutrition', {}).get('protein', 0) >= 20},
        'lean': {'breakfast': lambda m: m.get('nutrition', {}).get('carbs', 0) <= 25, 'lunch': lambda m: m.get('nutrition', {}).get('carbs', 0) <= 30, 'snack': lambda m: m.get('nutrition', {}).get('carbs', 0) <= 15},
        'weight_loss': {'breakfast': lambda m: m.get('calories', 0) <= 400, 'lunch': lambda m: m.get('calories', 0) <= 500, 'snack': lambda m: m.get('calories', 0) <= 200},
        'athletic': {'breakfast': lambda m: m.get('nutrition', {}).get('protein', 0) >= 22, 'lunch': lambda m: m.get('nutrition', {}).get('protein', 0) >= 28, 'snack': lambda m: m.get('nutrition', {}).get('protein', 0) >= 18},
        'maintain': {'breakfast': lambda m: True, 'lunch': lambda m: True, 'snack': lambda m: True}
    }
    filter_func = criteria.get(body_shape, {}).get(meal_type, lambda m: True)
    if not isinstance(meals, list): return []
    return [m for m in meals if isinstance(m, dict) and filter_func(m)]

def load_meals_from_dataframe(df, meal_type=None, nutrition_prefs=None, calorie_target=None, food_type_filter=None):
    essential_cols = ['calories', 'protein (g)', 'carbohydrates (g)', 'fat (g)', 'name_english']
    if not all(col in df.columns for col in essential_cols): return []
    df.dropna(subset=essential_cols, inplace=True)
    df['tags'] = df['tags'].fillna('unknown') if 'tags' in df.columns else 'unknown'
    if 'name_english' in df.columns: df.drop_duplicates(subset=['name_english'], inplace=True, keep='first')
    meals = []
    for index, row in df.iterrows():
        try:
            if pd.isna(row.get('name_english')): continue
            row_food_type = str(row.get('food_type', '')).lower()
            if food_type_filter and food_type_filter.lower() != 'mixed' and row_food_type != food_type_filter.lower(): continue
            calories = pd.to_numeric(row.get('calories'), errors='coerce'); protein = pd.to_numeric(row.get('protein (g)'), errors='coerce')
            carbs = pd.to_numeric(row.get('carbohydrates (g)'), errors='coerce'); fat = pd.to_numeric(row.get('fat (g)'), errors='coerce')
            if pd.isna(calories) or calories <= 0 or pd.isna(protein) or protein < 0 or pd.isna(carbs) or carbs < 0 or pd.isna(fat) or fat < 0: continue
            desc_en = str(row.get('description_english', '')); ingr_en = str(row.get('ingredients_english', '')); tags_str = str(row.get('tags', ''))
            combined_text = f"{desc_en} {ingr_en} {tags_str}".strip()
            meal_tags_raw = row.get('tags', ''); meal_tags = [tag.strip() for tag in str(meal_tags_raw).split(',')] if isinstance(meal_tags_raw, str) and meal_tags_raw else ['unknown']
            meal = {
                'id': f"{meal_type}-{index}", 'name_english': row['name_english'], 'name_arabic': row.get('name_arabic', ''),
                'calories': int(calories), 'meal_type': meal_type, 'food_type': row_food_type, 'combined_text': combined_text,
                'nutrition': {'protein': protein, 'carbs': carbs, 'fat': fat},
                'serving_size_english': str(row.get('serving_size_english', '')), 'serving_size_arabic': str(row.get('serving_size_arabic', '')),
                'bread_suggestion_english': str(row.get('bread_suggestion_english', '')), 'bread_suggestion_arabic': str(row.get('bread_suggestion_arabic', '')),
                'recipe_english': str(row.get('recipe_english', '')), 'recipe_arabic': str(row.get('recipe_arabic', '')),
                'tags': meal_tags, 'nutrition_score': 0
            }
            if nutrition_prefs and calorie_target is not None: meal['nutrition_score'] = calculate_nutrition_score(meal, nutrition_prefs, calorie_target / 3.0, meal_type)
            meals.append(meal)
        except Exception: pass
    return meals

def translate_meal(meal, language='english') -> Dict[str, Any]:
    if not isinstance(meal, dict):
        return MealOutput(id='error-invalid-input', name='Invalid Meal Data', calories=0, nutrition=MealNutritionOutput(protein=0, carbs=0, fat=0), serving_size='', bread_suggestion='', recipe='').dict()
    lang_suffix = 'arabic' if language == 'arabic' else 'english'; fallback_suffix = 'english'
    def get_field(key_base): return meal.get(f'{key_base}_{lang_suffix}') or meal.get(f'{key_base}_{fallback_suffix}') or ''
    nutrition_data = meal.get('nutrition', {});
    if not isinstance(nutrition_data, dict): nutrition_data = {'protein': 0, 'carbs': 0, 'fat': 0}
    output_data = {
        'id': meal.get('id', 'no-id'), 'name': get_field('name'), 'calories': meal.get('calories', 0),
        'nutrition': {'protein': round(nutrition_data.get('protein', 0), 1), 'carbs': round(nutrition_data.get('carbs', 0), 1), 'fat': round(nutrition_data.get('fat', 0), 1)},
        'serving_size': get_field('serving_size'), 'bread_suggestion': get_field('bread_suggestion'), 'recipe': get_field('recipe')
    }
    bread = output_data['bread_suggestion'].lower().strip()
    if bread in ['no bread suggested', 'لا يقترح خبز', '', 'none', 'n/a']: output_data['bread_suggestion'] = None
    if not output_data['serving_size']: output_data['serving_size'] = None
    if not output_data['recipe']: output_data['recipe'] = None
    try:
        validated_output = MealOutput(**output_data)
        return validated_output.dict()
    except Exception as e:
        return MealOutput(id=meal.get('id', 'error-validation'), name='Data Error', calories=0, nutrition=MealNutritionOutput(protein=0, carbs=0, fat=0), serving_size=None, bread_suggestion=None, recipe=None).dict()

def get_nutrient_weights(nutrition_prefs):
    weights = {'protein': 0.5, 'carbs': 0.5, 'fat': 0.5};
    if not isinstance(nutrition_prefs, dict): return weights
    preference_map = {'high': 1.5, 'moderate': 1.0, 'yes': 1.0, 'no preference': 0.5, 'low': -1.0, 'no': -1.0, 'very-low': -1.5}
    for nutrient, pref in nutrition_prefs.items():
        if nutrient in weights: weights[nutrient] = preference_map.get(str(pref).lower().strip(), 0.5)
    return weights

def normalize_nutrition(meals):
    if not meals: return []
    nutrition_data = [m.get('nutrition', {'protein': 0, 'carbs': 0, 'fat': 0}) for m in meals if isinstance(m, dict)]
    if not nutrition_data: return [[0, 0, 0]] * len(meals)
    df = pd.DataFrame(nutrition_data); df['protein'] = df.get('protein', 0); df['carbs'] = df.get('carbs', 0); df['fat'] = df.get('fat', 0)
    for col in ['protein', 'carbs', 'fat']:
        min_val, max_val = df[col].min(), df[col].max(); range_val = max_val - min_val
        df[col] = (df[col] - min_val) / range_val if range_val > 0 else 0
    df.fillna(0, inplace=True); return df[['protein', 'carbs', 'fat']].values.tolist()

def calculate_content_score(meal, weights, normalized_features, index_map):
    if not isinstance(meal, dict) or 'id' not in meal: return 0.0
    meal_id = meal['id']; idx = index_map.get(meal_id)
    if idx is None or idx >= len(normalized_features): return 0.0
    features = normalized_features[idx];
    if not isinstance(features, (list, np.ndarray)) or len(features) < 3: return 0.0
    w_prot = weights.get('protein', 0.5); w_carb = weights.get('carbs', 0.5); w_fat = weights.get('fat', 0.5)
    f_prot, f_carb, f_fat = features[0], features[1], features[2]
    return (w_prot * f_prot + w_carb * f_carb + w_fat * f_fat)

def create_user_meal_matrix(meal_lists, user_preferences, num_users=10):
    all_meals_list = [m for meals_in_type in meal_lists.values() if isinstance(meals_in_type, list) for m in meals_in_type if isinstance(m, dict) and 'id' in m]
    unique_meals_dict = {m['id']: m for m in all_meals_list if m.get('id')}
    all_meals = list(unique_meals_dict.values())
    if not all_meals: return csr_matrix((0, 0)), [], []
    meal_id_to_index = {meal['id']: i for i, meal in enumerate(all_meals)}
    meal_names = [meal.get('name_english', f'Unknown Meal {i}') for i, meal in enumerate(all_meals)]
    num_items = len(all_meals); rows, cols, data = [], [], []; real_user_rated_indices = set()
    if user_preferences and user_preferences[0]:
        ratings = user_preferences[0].get('ratings', {})
        for meal_id, rating in ratings.items():
            meal_idx = meal_id_to_index.get(meal_id)
            if meal_idx is not None and isinstance(rating, (int, float)) and 1 <= rating <= 5:
                rows.append(0); cols.append(meal_idx); data.append(float(rating)); real_user_rated_indices.add(meal_idx)
    for user_idx in range(1, num_users):
        num_ratings = random.randint(max(1, num_items // 10), max(2, num_items // 5))
        items_indices = random.sample(range(num_items), min(num_ratings, num_items))
        for meal_idx in items_indices:
            if meal_idx not in real_user_rated_indices: rows.append(user_idx); cols.append(meal_idx); data.append(float(random.randint(1, 5)))
    if not data: return csr_matrix((num_users, num_items)), all_meals, meal_names
    return csr_matrix((data, (rows, cols)), shape=(num_users, num_items)), all_meals, meal_names

def apply_svd(user_meal_matrix, n_components=5):
    if user_meal_matrix is None or not isinstance(user_meal_matrix, csr_matrix): return None, None, user_meal_matrix
    n_users, n_items = user_meal_matrix.shape
    if n_users < 2 or n_items < 2 or user_meal_matrix.nnz == 0: return None, None, user_meal_matrix
    effective_n_components = min(n_components, n_users - 1, n_items - 1)
    if effective_n_components < 1: return None, None, user_meal_matrix
    try:
        svd = TruncatedSVD(n_components=effective_n_components, random_state=42)
        user_factors = svd.fit_transform(user_meal_matrix)
        corr_matrix = None
        if user_factors.shape[0] >= 2 and user_factors.shape[1] > 0:
            with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning); corr_temp = np.corrcoef(user_factors)
            if not np.isnan(corr_temp).all(): corr_matrix = corr_temp
        return svd, corr_matrix, user_meal_matrix
    except Exception: return None, None, user_meal_matrix

def knn_recommendations(user_meal_matrix, all_meals, meal_names, user_index, n_neighbors=5, top_n=3):
    if user_meal_matrix is None or not isinstance(user_meal_matrix, csr_matrix): return []
    n_users, n_items = user_meal_matrix.shape
    if n_users <= user_index or n_items == 0 or user_meal_matrix.nnz == 0 or not all_meals: return []
    effective_n_neighbors = min(n_neighbors, n_users - 1);
    if effective_n_neighbors <= 0: return []
    try:
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute'); model_knn.fit(user_meal_matrix)
        distances, indices = model_knn.kneighbors(user_meal_matrix[user_index], n_neighbors=effective_n_neighbors + 1)
        distances, indices = distances.flatten(), indices.flatten()
    except Exception: return []
    recommendations = {}; user_rated_indices = set(user_meal_matrix[user_index].nonzero()[1]); start_index = 1 if indices[0] == user_index else 0
    for i in range(start_index, len(indices)):
        neighbor_idx = indices[i];
        if neighbor_idx >= n_users: continue
        neighbor_similarity = 1 - distances[i]; neighbor_ratings_sparse = user_meal_matrix[neighbor_idx]
        neighbor_rated_indices = neighbor_ratings_sparse.nonzero()[1]; neighbor_ratings_values = neighbor_ratings_sparse.data
        for item_idx, rating in zip(neighbor_rated_indices, neighbor_ratings_values):
            if item_idx not in user_rated_indices and rating > 3 and 0 <= item_idx < len(all_meals):
                meal = all_meals[item_idx]; meal_id = meal.get('id')
                if meal_id: score = neighbor_similarity * (rating - 3); recommendations[meal_id] = recommendations.get(meal_id, 0) + score
    sorted_recommendations = sorted(recommendations.items(), key=lambda item: item[1], reverse=True); final_recommendations = []
    seen_ids = set()
    for meal_id, score in sorted_recommendations:
        if len(final_recommendations) >= top_n: break
        if meal_id not in seen_ids:
            meal_obj = next((m for m in all_meals if m.get('id') == meal_id), None)
            if meal_obj: final_recommendations.append(meal_obj); seen_ids.add(meal_id)
    return final_recommendations

def calculate_macronutrient_similarity(meal1, meal2, weights):
    if not isinstance(meal1, dict) or not isinstance(meal2, dict): return 0.0
    macros1 = meal1.get('nutrition', {}); macros2 = meal2.get('nutrition', {})
    if not isinstance(macros1, dict) or not isinstance(macros2, dict): return 0.0
    p1, c1, f1 = macros1.get('protein', 0), macros1.get('carbs', 0), macros1.get('fat', 0)
    p2, c2, f2 = macros2.get('protein', 0), macros2.get('carbs', 0), macros2.get('fat', 0)
    if not isinstance(weights, dict): weights = {}
    w_prot, w_carb, w_fat = weights.get('protein', 0.33), weights.get('carbs', 0.33), weights.get('fat', 0.33)
    total_weight = w_prot + w_carb + w_fat
    if total_weight <= 0: w_prot, w_carb, w_fat, total_weight = 0.33, 0.33, 0.33, 1.0
    else: w_prot /= total_weight; w_carb /= total_weight; w_fat /= total_weight
    max_prot, max_carb, max_fat = max(p1, p2, 1), max(c1, c2, 1), max(f1, f2, 1)
    prot_sim = 1 - abs(p1 - p2) / max_prot; carb_sim = 1 - abs(c1 - c2) / max_carb; fat_sim = 1 - abs(f1 - f2) / max_fat
    similarity = w_prot * prot_sim + w_carb * carb_sim + w_fat * fat_sim; return max(0.0, similarity)

def tfidf_recommendations(meals, query_meal, top_n=3, tfidf_vectorizer=None, dynamic_weights=None):
    if not meals or not query_meal or tfidf_vectorizer is None or not hasattr(tfidf_vectorizer, 'vocabulary_') or not tfidf_vectorizer.vocabulary_: return []
    if dynamic_weights is None: dynamic_weights = {'tfidf_weight': 0.7, 'macro_weight': 0.3, 'macro_weights': {'protein': 0.4, 'carbs': 0.3, 'fat': 0.3}}
    try:
        documents = [meal.get('combined_text', '') for meal in meals]; query_text = query_meal.get('combined_text', ''); query_nutrition = query_meal.get('nutrition', {})
        enhanced_query = f"{query_text} protein_{query_nutrition.get('protein', 0):.0f} carbs_{query_nutrition.get('carbs', 0):.0f} fat_{query_nutrition.get('fat', 0):.0f}".strip()
        tfidf_matrix = tfidf_vectorizer.transform(documents); query_tfidf = tfidf_vectorizer.transform([enhanced_query])
        tfidf_sim = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    except Exception: return []
    macro_sim = np.array([calculate_macronutrient_similarity(query_meal, meal, dynamic_weights.get('macro_weights', {})) for meal in meals])
    tfidf_w = dynamic_weights.get('tfidf_weight', 0.7); macro_w = dynamic_weights.get('macro_weight', 0.3); total_w = tfidf_w + macro_w
    if total_w <= 0: tfidf_w, macro_w = 0.5, 0.5; 
    else: tfidf_w /= total_w; macro_w /= total_w
    hybrid_sim = (tfidf_w * tfidf_sim + macro_w * macro_sim)
    query_idx = -1; query_id = query_meal.get('id');
    if query_id: query_idx = next((i for i, meal in enumerate(meals) if meal.get('id') == query_id), -1)
    if query_idx != -1: hybrid_sim[query_idx] = -np.inf
    num_avail = len(meals) - (1 if query_idx != -1 else 0); actual_n = min(top_n, num_avail);
    if actual_n <= 0: return []
    sim_indices = np.argsort(hybrid_sim)[-actual_n:][::-1]; return [meals[i] for i in sim_indices if 0 <= i < len(meals)]

def get_meal_by_id(meal_id, meal_lists):
    if not meal_id or not isinstance(meal_lists, dict): return None
    for meal_list in meal_lists.values():
        if isinstance(meal_list, list):
            for meal in meal_list:
                if isinstance(meal, dict) and meal.get('id') == meal_id: return meal
    return None

def hybrid_recommendations(user_index, svd, corr_matrix, user_meal_matrix, all_meals, meal_names, calorie_target, user_preferences, top_n=3, knn_for_cold_start=True, tfidf_vector=None, dynamic_weights=None):
    global low_rated_meals, user_ratings
    if not all_meals or not user_preferences or not user_preferences[0]: return random.sample(all_meals, min(top_n, len(all_meals))) if all_meals else []
    user_prefs = user_preferences[0]; nutrition_prefs = user_prefs.get('nutrition_preferences', {'protein': 'moderate', 'carbs': 'moderate', 'fat': 'moderate'})
    if dynamic_weights is None: dynamic_weights = {'tfidf_weight': 0.7, 'macro_weight': 0.3, 'macro_weights': {'protein': 0.4, 'carbs': 0.3, 'fat': 0.3}}
    recommendations = {}; collab_recs_ids = set(); tfidf_recs_ids = set()
    num_rated = 0
    if user_meal_matrix is not None and user_meal_matrix.shape[0] > user_index:
        num_rated = user_meal_matrix[user_index].nnz
        if num_rated > 0 or knn_for_cold_start:
             collab_recs = knn_recommendations(user_meal_matrix, all_meals, meal_names, user_index, 10, top_n * 3); collab_w = 0.4
             for i, rec in enumerate(collab_recs):
                 mid = rec.get('id');
                 if mid and mid not in low_rated_meals: recommendations[mid] = recommendations.get(mid, 0) + collab_w * (1 / (i + 1)); collab_recs_ids.add(mid)
    if tfidf_vector is not None and hasattr(tfidf_vector, 'vocabulary_') and tfidf_vector.vocabulary_:
        high_rated = sorted([(rid, r) for rid, r in user_ratings.items() if r >= 4], key=lambda item: item[1], reverse=True)
        query_meal = get_meal_by_id(high_rated[0][0], {'all': all_meals}) if high_rated else None
        if query_meal:
            tfidf_recs = tfidf_recommendations(all_meals, query_meal, top_n * 3, tfidf_vector, dynamic_weights); tfidf_w_h = 0.3
            for i, rec in enumerate(tfidf_recs):
                mid = rec.get('id');
                if mid and mid not in low_rated_meals and mid != query_meal.get('id'): recommendations[mid] = recommendations.get(mid, 0) + tfidf_w_h * (1 / (i + 1)); tfidf_recs_ids.add(mid)
    weights = get_nutrient_weights(nutrition_prefs); normalized_features = []; index_map = {}
    try:
        normalized_features = normalize_nutrition(all_meals);
        if not normalized_features or len(normalized_features) != len(all_meals): raise ValueError()
        index_map = {m['id']: i for i, m in enumerate(all_meals) if m.get('id')}
    except Exception: pass
    if normalized_features:
        content_w = 0.3
        for meal in all_meals:
            mid = meal.get('id');
            if mid and mid not in low_rated_meals: score = calculate_content_score(meal, weights, normalized_features, index_map); recommendations[mid] = recommendations.get(mid, 0) + (content_w * score)
    final_scored = {mid: score for mid, score in recommendations.items() if mid not in low_rated_meals}
    sorted_recs = sorted(final_scored.items(), key=lambda item: item[1], reverse=True); final_list = []
    seen_ids = set()
    for mid, score in sorted_recs:
        if len(final_list) >= top_n: break
        if mid not in seen_ids:
            meal_obj = get_meal_by_id(mid, {'all': all_meals});
            if meal_obj: final_list.append(meal_obj); seen_ids.add(mid)
    if not final_list and all_meals:
         available = [m for m in all_meals if m.get('id') and m['id'] not in low_rated_meals]
         return random.sample(available, min(top_n, len(available))) if available else random.sample(all_meals, min(top_n, len(all_meals)))
    return final_list

def recommend_meals_ai(calorie_target, user_preferences, language='english', food_type='egyptian'):
    global low_rated_meals, user_dynamic_weights, tfidf_vectorizer_global, meal_lists_global, user_ratings
    no_meals_fallback_data = {'id': 'no-meal', 'name_english': 'No Suitable Meal Found', 'name_arabic': 'لم يتم العثور على وجبة مناسبة', 'calories': 0, 'nutrition': {'protein': 0, 'carbs': 0, 'fat': 0}, 'recipe_english': 'N/A', 'recipe_arabic': 'غير متوفر', 'meal_type': 'unknown', 'tags': [], 'serving_size_english': '', 'serving_size_arabic': '', 'bread_suggestion_english': '', 'bread_suggestion_arabic': ''}
    no_meals_fallback = translate_meal(no_meals_fallback_data, language)
    try: breakfast_df = pd.read_csv('breakfast.csv'); lunch_df = pd.read_csv('lunch.csv'); snack_df = pd.read_csv('snack.csv')
    except Exception as e: raise FileNotFoundError(f"Missing or error reading CSV file: {e}")
    user_prefs = user_preferences[0] if user_preferences else {}; nutrition_prefs = user_prefs.get('nutrition_preferences', {'body_shape': 'maintain'})
    bf_meals = load_meals_from_dataframe(breakfast_df, 'breakfast', nutrition_prefs, calorie_target, food_type); lu_meals = load_meals_from_dataframe(lunch_df, 'lunch', nutrition_prefs, calorie_target, food_type); sn_meals = load_meals_from_dataframe(snack_df, 'snack', nutrition_prefs, calorie_target, food_type)
    initial_lists = {'breakfast': bf_meals, 'lunch': lu_meals, 'snack': sn_meals}
    filtered_lists = {mtype: [m for m in mlist if isinstance(m, dict) and m.get('id') and m['id'] not in low_rated_meals] if isinstance(mlist, list) else [] for mtype, mlist in initial_lists.items()}
    body_shape = nutrition_prefs.get('body_shape')
    meal_lists = {mtype: filter_meals_by_body_shape(mlist, body_shape, mtype) for mtype, mlist in filtered_lists.items()} if body_shape else filtered_lists
    if not any(mlist for mlist in meal_lists.values()): meal_lists = filtered_lists
    meal_lists_global = meal_lists;
    if not any(v for v in meal_lists.values()): return [], []
    if 'ratings' not in user_prefs: user_prefs['ratings'] = user_ratings
    user_meal_matrix, all_meals, meal_names = create_user_meal_matrix(meal_lists, [user_prefs])
    if all_meals:
        docs = [m.get('combined_text', '') for m in all_meals];
        try:
            if any(doc and doc.strip() for doc in docs): tfidf_vectorizer_global = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=1); tfidf_vectorizer_global.fit(docs)
            else: tfidf_vectorizer_global = TfidfVectorizer(stop_words='english')
        except ValueError: tfidf_vectorizer_global = TfidfVectorizer(stop_words='english')
    else: tfidf_vectorizer_global = TfidfVectorizer(stop_words='english')
    svd_model, corr_matrix = None, None
    if user_meal_matrix is not None and user_meal_matrix.shape[0] >= 2 and user_meal_matrix.shape[1] > 1:
        n_comp = min(5, user_meal_matrix.shape[1]-1); svd_model, corr_matrix, _ = apply_svd(user_meal_matrix, n_components=n_comp)
    user_index = 0; weekly_plan_raw = []; days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']; days_ar = ['الاثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت', 'الأحد']
    plan_used_ids = set()
    for i in range(7):
        day_name = days[i] if language == 'english' else days_ar[i]; daily_meals = {}
        for meal_type in ['breakfast', 'lunch', 'snack']:
            dyn_weights = user_dynamic_weights[user_index]
            recs = hybrid_recommendations(user_index, svd_model, corr_matrix, user_meal_matrix, all_meals, meal_names, calorie_target, [user_prefs], 15, True, tfidf_vectorizer_global, dyn_weights)
            best_meal_data = next((rec for rec in recs if isinstance(rec, dict) and rec.get('meal_type') == meal_type and rec.get('id') and rec['id'] not in plan_used_ids), None)
            if best_meal_data is None:
                fallback = [m for m in meal_lists.get(meal_type, []) if isinstance(m, dict) and m.get('id') and m['id'] not in plan_used_ids];
                if fallback: fallback.sort(key=lambda m: m.get('nutrition_score', 0), reverse=True); best_meal_data = fallback[0]
                else:
                    reuse = meal_lists.get(meal_type, []);
                    if reuse: reuse.sort(key=lambda m: user_ratings.get(m.get('id', ''), 0), reverse=True); best_meal_data = reuse[0] if reuse else None
            chosen_meal_obj_data = best_meal_data if best_meal_data else no_meals_fallback_data
            mid = chosen_meal_obj_data.get('id');
            if mid and mid != 'no-meal': plan_used_ids.add(mid)
            daily_meals[meal_type] = translate_meal(chosen_meal_obj_data, language)
        weekly_plan_raw.append({'day': day_name, 'breakfast': daily_meals.get('breakfast', no_meals_fallback), 'lunch': daily_meals.get('lunch', no_meals_fallback), 'snack': daily_meals.get('snack', no_meals_fallback)})
    return weekly_plan_raw, all_meals

def retrain_model(user_preferences, meal_lists, all_meals_input):
    global user_ratings, tfidf_vectorizer_global
    if not user_preferences or not user_preferences[0]: return None, None, None
    if 'ratings' not in user_preferences[0]: user_preferences[0]['ratings'] = {}
    user_preferences[0]['ratings'].update(user_ratings.copy())
    user_meal_matrix, current_all_meals, meal_names = create_user_meal_matrix(meal_lists, user_preferences)
    if user_meal_matrix is None or user_meal_matrix.shape[0] < 1 or user_meal_matrix.shape[1] < 2: return None, None, None
    n_users, n_items = user_meal_matrix.shape; eff_n_comp = min(5, n_users - 1, n_items - 1)
    if eff_n_comp < 1: return None, None, user_meal_matrix
    svd, corr, matrix_out = apply_svd(user_meal_matrix, n_components=eff_n_comp)
    if svd is None: return None, None, matrix_out
    if current_all_meals:
        docs = [m.get('combined_text', '') for m in current_all_meals];
        try:
            if any(d and d.strip() for d in docs): tfidf_vectorizer_global = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=1); tfidf_vectorizer_global.fit(docs)
        except ValueError: pass
    return svd, corr, matrix_out

def update_dynamic_weights(user_id, weekly_plan):
    global user_ratings, user_dynamic_weights, meal_lists_global
    if not meal_lists_global or not weekly_plan or not user_ratings: return
    adj = 0.03; min_w = 0.05; max_w = 0.95; macro_min = 0.1; macro_max = 0.8
    weights = user_dynamic_weights[user_id];
    if 'macro_weights' not in weights or not isinstance(weights['macro_weights'], dict): weights['macro_weights'] = {'protein': 0.4, 'carbs': 0.3, 'fat': 0.3}
    macro_w = weights['macro_weights']
    n_rated = 0; p_adj, c_adj, f_adj, r_sum = 0.0, 0.0, 0.0, 0.0
    all_orig = {m['id']: m for mlist in meal_lists_global.values() if isinstance(mlist, list) for m in mlist if isinstance(m, dict) and m.get('id')}
    for day in weekly_plan:
        if not isinstance(day, dict): continue
        for meal_type in ['breakfast', 'lunch', 'snack']:
            meal = day.get(meal_type);
            if not isinstance(meal, dict): continue
            mid = meal.get('id')
            if mid and mid in user_ratings and mid in all_orig:
                rating = user_ratings[mid]; orig = all_orig[mid]; nutr = orig.get('nutrition');
                if not isinstance(nutr, dict): continue
                n_rated += 1; r_sum += rating; p, c, f = nutr.get('protein', 0), nutr.get('carbs', 0), nutr.get('fat', 0)
                r_diff = rating - 3.0; total_m = p + c + f;
                if total_m <= 0: continue
                p_adj += r_diff * (p / total_m); c_adj += r_diff * (c / total_m); f_adj += r_diff * (f / total_m)
    if n_rated > 0:
        avg_p, avg_c, avg_f = (p_adj / n_rated) * adj, (c_adj / n_rated) * adj, (f_adj / n_rated) * adj
        macro_w['protein'] = max(macro_min, min(macro_max, macro_w.get('protein', 0.33) + avg_p))
        macro_w['carbs'] = max(macro_min, min(macro_max, macro_w.get('carbs', 0.33) + avg_c))
        macro_w['fat'] = max(macro_min, min(macro_max, macro_w.get('fat', 0.33) + avg_f))
        total_mw_new = sum(macro_w.values());
        if total_mw_new > 0: macro_w['protein'] /= total_mw_new; macro_w['carbs'] /= total_mw_new; macro_w['fat'] /= total_mw_new
        else: macro_w = {'protein': 0.4, 'carbs': 0.3, 'fat': 0.3}
        avg_r = r_sum / n_rated; curr_macro_w = weights.get('macro_weight', 0.3)
        if avg_r > 3.5: weights['macro_weight'] = min(max_w, curr_macro_w + adj / 2)
        elif avg_r < 2.5: weights['macro_weight'] = max(min_w, curr_macro_w - adj / 2)
        weights['macro_weight'] = max(min_w, min(max_w, weights.get('macro_weight', 0.3)))
        weights['tfidf_weight'] = 1.0 - weights['macro_weight']; weights['macro_weights'] = macro_w
        user_dynamic_weights[user_id] = weights


@app.post("/set_user_info", response_model=UserInfoOutput)
async def set_user_info(user_input: UserInfoInput):
    session_id = str(uuid.uuid4())
    session = SessionState(); sessions[session_id] = session
    if user_input.language not in ['english', 'arabic']: raise HTTPException(status_code=400, detail="Invalid language.")
    if user_input.food_type not in ['egyptian', 'american', 'mixed']: raise HTTPException(status_code=400, detail="Invalid food_type.")
    if user_input.gender.lower() not in ['male', 'female', 'ذكر', 'أنثى']: raise HTTPException(status_code=400, detail="Invalid gender.")
    activity_level = user_input.activity_level.lower()
    if activity_level not in ACTIVITY_LEVELS: raise HTTPException(status_code=400, detail=f"Invalid activity_level.")
    body_shape_goal = user_input.body_shape_goal.lower()
    if body_shape_goal not in BODY_SHAPES: raise HTTPException(status_code=400, detail=f"Invalid body_shape_goal.")
    session.language = user_input.language; session.food_type = user_input.food_type
    session.weight = user_input.weight; session.height = user_input.height; session.age = user_input.age
    session.gender = user_input.gender; session.activity_level = activity_level
    try:
        session.bmr = calculate_bmr(session.weight, session.height, session.age, session.gender)
        session.tdee = calculate_tdee(session.bmr, session.activity_level)
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
    body_shape_info = BODY_SHAPES[body_shape_goal]
    session.user_preferences = [{'preferences': [], 'preferred_cuisine': session.food_type, 'nutrition_prefs': body_shape_info['nutrition_prefs'], 'ratings': {}}]
    calorie_adjustment = body_shape_info['calorie_factor']
    goal_adjusted_tdee = session.tdee * calorie_adjustment
    if body_shape_goal == 'weight_loss': session.calorie_target = goal_adjusted_tdee - 500
    elif body_shape_goal == 'muscular': session.calorie_target = goal_adjusted_tdee + 300
    else: session.calorie_target = goal_adjusted_tdee
    session.calorie_target = int(max(1200, session.calorie_target))
    height_m = session.height / 100; bmi = session.weight / (height_m ** 2) if height_m > 0 else 0
    min_healthy = 18.5 * (height_m ** 2) if height_m > 0 else 0; max_healthy = 24.9 * (height_m ** 2) if height_m > 0 else 0
    to_lose = max(0, session.weight - max_healthy); to_gain = max(0, min_healthy - session.weight)
    bmi_status = f"Estimated Daily Calorie Target: {session.calorie_target} calories\n"
    bmi_status += f"Current BMI: {bmi:.1f} (Healthy range: 18.5-24.9)\n"
    if to_lose > 0: bmi_status += f"To reach healthy BMI, potential weight loss: {to_lose:.1f} kg"
    elif to_gain > 0: bmi_status += f"To reach healthy BMI, potential weight gain: {to_gain:.1f} kg"
    else: bmi_status += "You are within a healthy weight range!"
    return UserInfoOutput(session_id=session_id, bmr=round(session.bmr,1), tdee=round(session.tdee,1), calorie_target=session.calorie_target, bmi_status=bmi_status)

@app.get("/generate_plan", response_model=WeeklyPlanOutput)
async def generate_plan(session_id: str):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    session: SessionState = sessions[session_id]
    if session.calorie_target is None or session.user_preferences is None: raise HTTPException(status_code=400, detail="User info must be set first via /set_user_info")
    try:
        weekly_plan_raw, current_all_meals = recommend_meals_ai(session.calorie_target, session.user_preferences, session.language, session.food_type)
        session.weekly_plan = weekly_plan_raw
        session.last_plan_rated_positively = False
    except FileNotFoundError as e: raise HTTPException(status_code=500, detail=f"Server configuration error: Missing data file {e.filename}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
    session.plan_count += 1
    if not session.weekly_plan: raise HTTPException(status_code=404, detail="Could not generate plan.")
    return {"weekly_plan": session.weekly_plan}

@app.post("/plan_feedback")
async def plan_feedback(session_id: str, feedback: FeedbackInput):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    session: SessionState = sessions[session_id]
    session.last_plan_rated_positively = feedback.satisfied
    if feedback.satisfied:
        message = "Great! Enjoy your meals. Remember consistency is key. Thank you for using Fat2Fit!"
        if session.language == 'arabic': message = "!رائع! استمتع بوجباتك. تذكر أن الاستمرارية هي المفتاح. شكرًا لاستخدامك Fat2Fit"
        return {"message": message}
    else:
        message = "Okay, feedback noted. If you'd like to provide specific meal ratings to improve the next plan, please use the /submit_ratings endpoint before regenerating."
        if session.language == 'arabic': message = ".حسنًا ، تم تدوين الملاحظات. إذا كنت ترغب في تقديم تقييمات وجبات محددة لتحسين الخطة التالية ، فيرجى استخدام نقطة النهاية /submit_ratings قبل إعادة الإنشاء"
        return {"message": message}

@app.post("/submit_ratings")
async def submit_ratings_api(session_id: str, ratings_input: list[Rating]):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    session: SessionState = sessions[session_id]
    global user_ratings, all_rated_meals, low_rated_meals, meal_lists_global
    ratings_processed = 0
    for rating in ratings_input:
        if not (1 <= rating.rating <= 5): continue
        meal_id = rating.meal_id; user_ratings[meal_id] = rating.rating; all_rated_meals.add(meal_id)
        if rating.rating <= 2: low_rated_meals.add(meal_id)
        elif meal_id in low_rated_meals: low_rated_meals.discard(meal_id)
        ratings_processed += 1
    if ratings_processed == 0: return {"status": "No valid ratings provided."}
    user_id_for_globals = 0
    try:
        update_dynamic_weights(user_id_for_globals, session.weekly_plan if session.weekly_plan else [])
        current_prefs = session.user_preferences if session.user_preferences else [{'ratings': {}}]
        if current_prefs and current_prefs[0]: current_prefs[0]['ratings'] = user_ratings.copy()
        else: current_prefs = [{'ratings': user_ratings.copy()}]
        retrain_model(current_prefs, meal_lists_global, [])
    except Exception as e: raise HTTPException(status_code=500, detail=f"Ratings submitted, but error during model update: {e}")
    return {"status": f"{ratings_processed} ratings updated successfully. Model adapted."}

@app.get("/regenerate_plan", response_model=WeeklyPlanOutput)
async def regenerate_plan(session_id: str):
    if session_id not in sessions: raise HTTPException(status_code=404, detail="Session not found")
    session: SessionState = sessions[session_id]
    if session.calorie_target is None or session.user_preferences is None: raise HTTPException(status_code=400, detail="Cannot regenerate plan. User info must be set first.")
    if not session.weekly_plan: raise HTTPException(status_code=400, detail="Generate a plan first before regenerating.")
    try:
        weekly_plan_raw, _ = recommend_meals_ai(session.calorie_target, session.user_preferences, session.language, session.food_type)
        session.weekly_plan = weekly_plan_raw
        session.last_plan_rated_positively = False
    except FileNotFoundError as e: raise HTTPException(status_code=500, detail=f"Server error: Missing data file {e.filename}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"Internal error during regeneration: {e}")
    session.plan_count += 1
    if not session.weekly_plan: raise HTTPException(status_code=404, detail="Could not regenerate plan.")
    return {"weekly_plan": session.weekly_plan}
