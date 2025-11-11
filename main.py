from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uuid
from typing import Optional, Dict, List
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from openai import OpenAI
import json
import uvicorn
import os
from datetime import datetime, timedelta
from urllib.parse import urlparse
from enum import Enum

app = FastAPI(title="Google Ads AI Automation - Full RMF Demo")
templates = Jinja2Templates(directory="templates")

# Hardcoded login credentials for demo
VALID_USERNAME = "google-review@growthsynth.com"
VALID_PASSWORD = "DemoAccess2024!"

# --- DIRECT CONFIGURATION (FOR DEMO PURPOSES) ---
GOOGLE_CLIENT_ID = "1071808823811-2n6bvfv1hciui0de5am7bohl1pl2kepe.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
GOOGLE_DEVELOPER_TOKEN = "o24jap6s20DTvbozGX_Z_g"
GOOGLE_ADS_REFRESH_TOKEN = "1//0gyK0R-OXGU-oCgYIARAAGBASNwF-L9Irr8WpHMNuUgBHQsrJ9H2kW7awaepyZULrGtli4w0IJAmMFJKYtYKMrKKMVN1ohWkAaw8"
GOOGLE_ADS_LOGIN_CUSTOMER_ID = "6227620999"
GOOGLE_ADS_TARGET_CUSTOMER_ID = "8898565768"

# Cerebras AI Configuration
CEREBRAS_API_KEY = "csk-99rvw8jpx6pppcwwxj2ncjyxcp863vcxjhcjvfpmf9d644nh"
CEREBRAS_MODEL = "gpt-oss-120b"

# Initialize Database
class Database:
    pass

db = Database()

class CampaignObjective(str, Enum):
    SALES = "SALES"
    LEADS = "LEADS"
    WEBSITE_TRAFFIC = "WEBSITE_TRAFFIC"

class SimpleCampaignRequest(BaseModel):
    product_url: str
    campaign_name: str
    objective: CampaignObjective = CampaignObjective.WEBSITE_TRAFFIC
    daily_budget: float = 100.0
    product_description: Optional[str] = None

class CreateAccountRequest(BaseModel):
    business_name: str
    currency_code: str
    time_zone: str
    consent_acknowledged: bool

class UpdateCampaignStatusRequest(BaseModel):
    status: str

class SmartCampaignConfig:
    OBJECTIVE_CONFIGS = {
        CampaignObjective.SALES: {
            "bidding_strategy": "MAXIMIZE_CLICKS",
            "recommended_budget": 150.0,
            "enable_search_partners": True,
            "enable_display_network": False,
            "description": "Optimized for driving sales and revenue",
            "ad_focus": "conversions and sales",
            "keyword_intent": "high commercial intent"
        },
        CampaignObjective.LEADS: {
            "bidding_strategy": "MAXIMIZE_CLICKS",
            "recommended_budget": 100.0,
            "enable_search_partners": True,
            "enable_display_network": False,
            "description": "Optimized for generating quality leads",
            "ad_focus": "lead generation and sign-ups",
            "keyword_intent": "information and consideration"
        },
        CampaignObjective.WEBSITE_TRAFFIC: {
            "bidding_strategy": "MAXIMIZE_CLICKS",
            "recommended_budget": 75.0,
            "enable_search_partners": True,
            "enable_display_network": False,
            "description": "Optimized for driving website traffic",
            "ad_focus": "clicks and traffic",
            "keyword_intent": "broad awareness"
        }
    }
    MIN_DAILY_BUDGET = 50.0
    MAX_CPC_BID_LIMIT = 5.0

    @staticmethod
    def get_objective_config(objective: CampaignObjective) -> Dict:
        return SmartCampaignConfig.OBJECTIVE_CONFIGS.get(
            objective,
            SmartCampaignConfig.OBJECTIVE_CONFIGS[CampaignObjective.WEBSITE_TRAFFIC]
        )

    @staticmethod
    def get_location_from_url(url: str) -> List[str]:
        url_lower = url.lower()
        if '.uk' in url_lower or '.co.uk' in url_lower:
            return ["GB"]
        elif '.ca' in url_lower:
            return ["CA"]
        elif '.au' in url_lower:
            return ["AU"]
        elif '.de' in url_lower:
            return ["DE"]
        elif '.fr' in url_lower:
            return ["FR"]
        elif '.in' in url_lower:
            return ["IN"]
        return ["US"]

    @staticmethod
    def get_optimal_budget(user_budget: float, objective: CampaignObjective) -> float:
        objective_config = SmartCampaignConfig.get_objective_config(objective)
        recommended_budget = objective_config["recommended_budget"]
        if user_budget < SmartCampaignConfig.MIN_DAILY_BUDGET:
            print(f"⚠️ Budget ${user_budget} is low. Recommending minimum ${SmartCampaignConfig.MIN_DAILY_BUDGET}")
            return SmartCampaignConfig.MIN_DAILY_BUDGET
        if user_budget < recommended_budget:
            print(f"💡 For {objective.value} objective, recommended budget is ${recommended_budget}")
        return user_budget

class GoogleAdsAutomation:
    def __init__(self):
        self.cerebras_client = OpenAI(api_key=CEREBRAS_API_KEY, base_url="https://api.cerebras.ai/v1")
        self.google_ads_client = self._init_google_ads_client()

    def _init_google_ads_client(self):
        credentials = {
            "developer_token": GOOGLE_DEVELOPER_TOKEN,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": GOOGLE_ADS_REFRESH_TOKEN,
            "login_customer_id": GOOGLE_ADS_LOGIN_CUSTOMER_ID,
            "use_proto_plus": True
        }
        return GoogleAdsClient.load_from_dict(credentials)

    def create_google_ads_account(self, business_name: str, currency_code: str, time_zone: str) -> Dict:
        try:
            customer_service = self.google_ads_client.get_service("CustomerService")
            customer = self.google_ads_client.get_type("Customer")
            customer.descriptive_name = business_name
            customer.currency_code = currency_code
            customer.time_zone = time_zone
            response = customer_service.create_customer_client(
                customer_id=GOOGLE_ADS_LOGIN_CUSTOMER_ID,
                customer_client=customer
            )
            new_customer_id = response.resource_name.split('/')[-1]
            print(f"✅ Successfully created Google Ads account: {new_customer_id}")
            return {
                "success": True,
                "customer_id": new_customer_id,
                "business_name": business_name,
                "currency_code": currency_code,
                "time_zone": time_zone,
                "resource_name": response.resource_name,
                "message": "Account created successfully. Please log in to Google Ads to set up billing."
            }
        except GoogleAdsException as ex:
            error_msg = "Failed to create account: "
            if ex.failure and ex.failure.errors:
                for error in ex.failure.errors:
                    error_msg += f"{error.message}; "
            else:
                error_msg += str(ex)
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            raise Exception(str(e))

    def check_account_type(self, customer_id: str) -> Dict:
        try:
            ga_service = self.google_ads_client.get_service("GoogleAdsService")
            query = """
                SELECT
                    customer.descriptive_name,
                    customer.currency_code,
                    customer.time_zone,
                    customer.test_account,
                    customer.manager,
                    customer.id
                FROM customer
                LIMIT 1
            """
            request = self.google_ads_client.get_type("SearchGoogleAdsRequest")
            request.customer_id = customer_id
            request.query = query
            response = ga_service.search(request=request)
            customer = None
            for row in response:
                customer = row.customer
                break
            if not customer:
                return None
            is_test = customer.test_account
            account_info = {
                "customer_id": customer_id,
                "descriptive_name": customer.descriptive_name,
                "currency_code": customer.currency_code,
                "time_zone": customer.time_zone,
                "is_test_account": is_test,
                "is_manager": customer.manager,
                "account_type": "🧪 TEST ACCOUNT" if is_test else "🚀 PRODUCTION ACCOUNT",
                "can_serve_real_ads": not is_test,
                "needs_billing": not is_test,
                "status": "Active"
            }
            if is_test:
                account_info["recommendation"] = "This is a TEST account. Campaigns won't serve real ads. Create a production account at ads.google.com to run real campaigns."
            else:
                account_info["recommendation"] = "This is a PRODUCTION account. Add billing information to serve real ads and track performance."
            return account_info
        except GoogleAdsException as ex:
            print(f"❌ Error checking account: {ex}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return None

    def update_campaign_status(self, customer_id: str, campaign_id: str, new_status: str) -> Dict:
        try:
            campaign_service = self.google_ads_client.get_service("CampaignService")
            campaign_operation = self.google_ads_client.get_type("CampaignOperation")
            campaign = campaign_operation.update
            campaign.resource_name = campaign_service.campaign_path(customer_id, campaign_id)
            status_enum = self.google_ads_client.enums.CampaignStatusEnum
            if new_status.upper() == "PAUSED":
                campaign.status = status_enum.PAUSED
            elif new_status.upper() == "ENABLED":
                campaign.status = status_enum.ENABLED
            else:
                raise ValueError("Invalid status. Must be 'ENABLED' or 'PAUSED'.")
            campaign_operation.update_mask.paths.append("status")
            response = campaign_service.mutate_campaigns(
                customer_id=customer_id,
                operations=[campaign_operation]
            )
            updated_resource_name = response.results[0].resource_name
            print(f"✅ Campaign {campaign_id} status updated to {new_status}")
            return {
                "success": True,
                "resource_name": updated_resource_name,
                "new_status": new_status
            }
        except GoogleAdsException as ex:
            print(f"❌ Error updating campaign status: {ex}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error during status update: {e}")
            raise

    def get_all_campaigns(self, customer_id: str) -> List[Dict]:
        try:
            ga_service = self.google_ads_client.get_service("GoogleAdsService")
            query = """
                SELECT
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign.advertising_channel_type,
                    campaign_budget.amount_micros
                FROM campaign
                ORDER BY campaign.id DESC
            """
            request = self.google_ads_client.get_type("SearchGoogleAdsRequest")
            request.customer_id = customer_id
            request.query = query
            response = ga_service.search(request=request)
            campaigns = []
            for row in response:
                campaign = row.campaign
                budget = row.campaign_budget if hasattr(row, 'campaign_budget') else None
                campaigns.append({
                    "id": str(campaign.id),
                    "name": campaign.name,
                    "status": campaign.status.name,
                    "type": campaign.advertising_channel_type.name,
                    "budget": budget.amount_micros / 1000000 if budget else 0,
                    "resource_name": campaign.resource_name
                })
            return campaigns
        except GoogleAdsException as ex:
            print(f"❌ Error fetching campaigns: {ex}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return []

    def get_campaign_performance(self, customer_id: str, campaign_id: str, date_range: str = "LAST_30_DAYS") -> Dict:
        try:
            ga_service = self.google_ads_client.get_service("GoogleAdsService")
            query = f"""
                SELECT
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.cost_micros,
                    metrics.conversions
                FROM campaign
                WHERE campaign.id = {campaign_id}
                AND segments.date DURING {date_range}
            """
            request = self.google_ads_client.get_type("SearchGoogleAdsRequest")
            request.customer_id = customer_id
            request.query = query
            response = ga_service.search(request=request)
            total_impressions = 0
            total_clicks = 0
            total_cost = 0
            total_conversions = 0
            campaign_name = ""
            campaign_status = ""
            for row in response:
                if not campaign_name:
                    campaign_name = row.campaign.name
                    campaign_status = row.campaign.status.name
                total_impressions += row.metrics.impressions
                total_clicks += row.metrics.clicks
                total_cost += row.metrics.cost_micros
                total_conversions += row.metrics.conversions
            return {
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "campaign_status": campaign_status,
                "impressions": total_impressions,
                "clicks": total_clicks,
                "cost": total_cost / 1000000,
                "conversions": total_conversions,
                "date_range": date_range
            }
        except GoogleAdsException as ex:
            print(f"❌ Error fetching campaign performance: {ex}")
            return {
                "campaign_id": campaign_id,
                "error": "Could not fetch performance data",
                "impressions": 0,
                "clicks": 0,
                "cost": 0,
                "conversions": 0
            }

    def get_keyword_performance(self, customer_id: str, campaign_id: str, date_range: str = "LAST_30_DAYS") -> List[Dict]:
        try:
            ga_service = self.google_ads_client.get_service("GoogleAdsService")
            query = f"""
                SELECT
                    ad_group_criterion.keyword.text,
                    ad_group_criterion.keyword.match_type,
                    ad_group_criterion.status,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.cost_micros,
                    metrics.conversions
                FROM keyword_view
                WHERE campaign.id = {campaign_id}
                AND segments.date DURING {date_range}
                ORDER BY metrics.impressions DESC
                LIMIT 50
            """
            request = self.google_ads_client.get_type("SearchGoogleAdsRequest")
            request.customer_id = customer_id
            request.query = query
            response = ga_service.search(request=request)
            keywords = []
            for row in response:
                keywords.append({
                    "keyword": row.ad_group_criterion.keyword.text,
                    "match_type": row.ad_group_criterion.keyword.match_type.name,
                    "status": row.ad_group_criterion.status.name,
                    "impressions": row.metrics.impressions,
                    "clicks": row.metrics.clicks,
                    "cost": row.metrics.cost_micros / 1000000,
                    "conversions": row.metrics.conversions
                })
            return keywords
        except GoogleAdsException as ex:
            print(f"❌ Error fetching keyword performance: {ex}")
            return []

    def get_ad_performance(self, customer_id: str, campaign_id: str, date_range: str = "LAST_30_DAYS") -> List[Dict]:
        try:
            ga_service = self.google_ads_client.get_service("GoogleAdsService")
            query = f"""
                SELECT
                    ad_group_ad.ad.id,
                    ad_group_ad.ad.type,
                    ad_group_ad.status,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.cost_micros,
                    metrics.conversions
                FROM ad_group_ad
                WHERE campaign.id = {campaign_id}
                AND segments.date DURING {date_range}
                ORDER BY metrics.impressions DESC
                LIMIT 20
            """
            request = self.google_ads_client.get_type("SearchGoogleAdsRequest")
            request.customer_id = customer_id
            request.query = query
            response = ga_service.search(request=request)
            ads = []
            for row in response:
                ads.append({
                    "ad_id": str(row.ad_group_ad.ad.id),
                    "ad_type": row.ad_group_ad.ad.type_.name,
                    "status": row.ad_group_ad.status.name,
                    "impressions": row.metrics.impressions,
                    "clicks": row.metrics.clicks,
                    "cost": row.metrics.cost_micros / 1000000,
                    "conversions": row.metrics.conversions
                })
            return ads
        except GoogleAdsException as ex:
            print(f"❌ Error fetching ad performance: {ex}")
            return []

    def generate_keywords_with_ai(self, product_url: str, objective: CampaignObjective, product_description: str = None) -> Dict:
        domain = urlparse(product_url).netloc
        objective_config = SmartCampaignConfig.get_objective_config(objective)
        prompt = f"""Analyze: {product_url}
{product_description if product_description else ""}
Campaign Objective: {objective.value}
Focus: {objective_config['ad_focus']}
Keyword Intent: {objective_config['keyword_intent']}
Generate Google Ads campaign data optimized for {objective.value}.
STRICT RULES:
- Headlines: Max 30 characters
- Descriptions: Max 90 characters
- Keywords: Focus on {objective_config['keyword_intent']}
- Return ONLY valid JSON
- No markdown, no code blocks
Format:
{{
 "keywords": [
 {{"keyword": "buy product", "match_type": "BROAD"}},
 {{"keyword": "best deals", "match_type": "PHRASE"}}
 ],
 "headlines": [
 "Buy Now Save Big",
 "Top Quality"
 ],
 "descriptions": [
 "Get the best deals. Shop now and save!"
 ],
 "negative_keywords": ["free", "cheap"]
}}
Return JSON only.
"""
        try:
            response = self.cerebras_client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. No markdown. No explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            ai_response = response.choices[0].message.content.strip()
            if "```json" in ai_response:
                ai_response = ai_response.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_response:
                ai_response = ai_response.split("```")[1].split("```")[0].strip()
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found")
            json_str = ai_response[start_idx:end_idx]
            parsed_data = json.loads(json_str)
            parsed_data.setdefault('keywords', [
                {"keyword": f"{domain}", "match_type": "BROAD"},
                {"keyword": f"buy {domain}", "match_type": "PHRASE"}
            ])
            parsed_data.setdefault('headlines', ["Shop Now & Save", "Top Quality", "Best Deals"])
            parsed_data.setdefault('descriptions', [
                "Get the best deals on quality products. Shop now!",
                "Fast shipping and easy returns. Order today!"
            ])
            parsed_data.setdefault('negative_keywords', ["free", "cheap"])
            parsed_data['headlines'] = [h[:30] for h in parsed_data['headlines'][:15]]
            parsed_data['descriptions'] = [d[:90] for d in parsed_data['descriptions'][:5]]
            for kw in parsed_data['keywords']:
                if 'match_type' not in kw or kw['match_type'] not in ['BROAD', 'PHRASE', 'EXACT']:
                    kw['match_type'] = 'BROAD'
            print(f"✅ Generated {len(parsed_data['keywords'])} keywords for {objective.value}")
            return parsed_data
        except Exception as e:
            print(f"⚠️ AI generation error: {str(e)}, using fallback data")
            return {
                "keywords": [
                    {"keyword": f"{domain}", "match_type": "BROAD"},
                    {"keyword": f"buy from {domain}", "match_type": "PHRASE"},
                    {"keyword": "shop online", "match_type": "BROAD"}
                ],
                "headlines": ["Shop Now & Save", "Top Quality", "Best Deals Online"],
                "descriptions": [
                    "Get the best deals on quality products. Shop now and save!",
                    "Fast shipping and easy returns. Order today!"
                ],
                "negative_keywords": ["free", "cheap"]
            }

    def create_smart_campaign(self, request: SimpleCampaignRequest):
        campaign_service = self.google_ads_client.get_service("CampaignService")
        campaign_budget_service = self.google_ads_client.get_service("CampaignBudgetService")
        unique_id = str(uuid.uuid4()).split('-')[0]
        unique_campaign_name = f"{request.campaign_name} [{request.objective.value}] {unique_id}"
        objective_config = SmartCampaignConfig.get_objective_config(request.objective)
        optimal_budget = SmartCampaignConfig.get_optimal_budget(request.daily_budget, request.objective)
        print(f"\n{'='*70}\n🚀 CREATING CAMPAIGN - {request.objective.value}\n{'='*70}")
        budget_operation = self.google_ads_client.get_type("CampaignBudgetOperation")
        budget = budget_operation.create
        budget.name = f"{unique_campaign_name}_budget"
        budget.amount_micros = int(optimal_budget * 1000000)
        budget.delivery_method = self.google_ads_client.enums.BudgetDeliveryMethodEnum.STANDARD
        budget.explicitly_shared = False
        try:
            budget_response = campaign_budget_service.mutate_campaign_budgets(
                customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
                operations=[budget_operation]
            )
            budget_resource_name = budget_response.results[0].resource_name
            print(f"✅ Budget created")
        except GoogleAdsException as ex:
            print(f"❌ Budget creation failed")
            raise
        campaign_operation = self.google_ads_client.get_type("CampaignOperation")
        campaign = campaign_operation.create
        campaign.name = unique_campaign_name
        campaign.campaign_budget = budget_resource_name
        campaign.status = self.google_ads_client.enums.CampaignStatusEnum.PAUSED
        campaign.advertising_channel_type = self.google_ads_client.enums.AdvertisingChannelTypeEnum.SEARCH
        bidding_strategy = objective_config['bidding_strategy']
        if bidding_strategy == "MAXIMIZE_CLICKS":
            campaign.target_spend.cpc_bid_ceiling_micros = int(SmartCampaignConfig.MAX_CPC_BID_LIMIT * 1000000)
        else:
            campaign.manual_cpc.enhanced_cpc_enabled = True
        campaign.network_settings.target_google_search = True
        campaign.network_settings.target_search_network = objective_config['enable_search_partners']
        campaign.network_settings.target_content_network = objective_config['enable_display_network']
        campaign.network_settings.target_partner_search_network = False
        campaign.contains_eu_political_advertising = self.google_ads_client.enums.EuPoliticalAdvertisingStatusEnum.DOES_NOT_CONTAIN_EU_POLITICAL_ADVERTISING
        try:
            response = campaign_service.mutate_campaigns(
                customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
                operations=[campaign_operation]
            )
            created_campaign_resource = response.results[0].resource_name
            campaign_id = created_campaign_resource.split('/')[-1]
            print(f"✅ Campaign created: {campaign_id}")
            return created_campaign_resource, unique_campaign_name, optimal_budget, campaign_id
        except GoogleAdsException as ex:
            print(f"❌ Campaign creation failed")
            raise

    def add_smart_geo_targeting(self, campaign_resource: str, product_url: str):
        campaign_criterion_service = self.google_ads_client.get_service("CampaignCriterionService")
        locations = SmartCampaignConfig.get_location_from_url(product_url)
        location_ids = {
            "US": 2840,
            "CA": 2124,
            "GB": 2826,
            "AU": 2036,
            "DE": 2276,
            "FR": 2250,
            "IN": 2356
        }
        operations = []
        for country in locations:
            if country in location_ids:
                operation = self.google_ads_client.get_type("CampaignCriterionOperation")
                criterion = operation.create
                criterion.campaign = campaign_resource
                criterion.location.geo_target_constant = f"geoTargetConstants/{location_ids[country]}"
                operations.append(operation)
        if operations:
            try:
                campaign_criterion_service.mutate_campaign_criteria(
                    customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
                    operations=operations
                )
                print(f"✅ Added geo targeting: {locations}")
            except GoogleAdsException as ex:
                print(f"⚠️ Geo targeting warning")

    def add_smart_language_targeting(self, campaign_resource: str):
        campaign_criterion_service = self.google_ads_client.get_service("CampaignCriterionService")
        operation = self.google_ads_client.get_type("CampaignCriterionOperation")
        criterion = operation.create
        criterion.campaign = campaign_resource
        criterion.language.language_constant = "languageConstants/1000"
        try:
            campaign_criterion_service.mutate_campaign_criteria(
                customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
                operations=[operation]
            )
            print(f"✅ Added language targeting")
        except GoogleAdsException as ex:
            print(f"⚠️ Language targeting warning")

    def create_ad_group(self, campaign_resource: str, ad_group_name: str, cpc_bid: float = 1.0):
        ad_group_service = self.google_ads_client.get_service("AdGroupService")
        ad_group_operation = self.google_ads_client.get_type("AdGroupOperation")
        ad_group = ad_group_operation.create
        ad_group.name = ad_group_name
        ad_group.campaign = campaign_resource
        ad_group.type_ = self.google_ads_client.enums.AdGroupTypeEnum.SEARCH_STANDARD
        ad_group.cpc_bid_micros = int(cpc_bid * 1000000)
        ad_group.status = self.google_ads_client.enums.AdGroupStatusEnum.ENABLED
        response = ad_group_service.mutate_ad_groups(
            customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
            operations=[ad_group_operation]
        )
        return response.results[0].resource_name

    def add_keywords(self, ad_group_resource: str, keywords: List[Dict]):
        ad_group_criterion_service = self.google_ads_client.get_service("AdGroupCriterionService")
        successful_keywords = []
        for kw in keywords:
            try:
                operation = self.google_ads_client.get_type("AdGroupCriterionOperation")
                criterion = operation.create
                criterion.ad_group = ad_group_resource
                criterion.status = self.google_ads_client.enums.AdGroupCriterionStatusEnum.ENABLED
                criterion.keyword.text = kw['keyword']
                match_type_map = {
                    "BROAD": self.google_ads_client.enums.KeywordMatchTypeEnum.BROAD,
                    "PHRASE": self.google_ads_client.enums.KeywordMatchTypeEnum.PHRASE,
                    "EXACT": self.google_ads_client.enums.KeywordMatchTypeEnum.EXACT
                }
                criterion.keyword.match_type = match_type_map.get(kw['match_type'], self.google_ads_client.enums.KeywordMatchTypeEnum.BROAD)
                ad_group_criterion_service.mutate_ad_group_criteria(
                    customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
                    operations=[operation]
                )
                successful_keywords.append(kw['keyword'])
            except GoogleAdsException:
                continue
        print(f"✅ Added {len(successful_keywords)}/{len(keywords)} keywords")
        if len(successful_keywords) == 0:
            raise Exception("Could not add any keywords")

    def create_ad(self, ad_group_resource: str, headlines: List[str], descriptions: List[str], final_url: str):
        # Validation
        if len(headlines) < 3:
            raise ValueError(f"Minimum 3 headlines required. You provided {len(headlines)}.")
        if len(descriptions) < 2:
            raise ValueError(f"Minimum 2 descriptions required. You provided {len(descriptions)}.")
        
        ad_group_ad_service = self.google_ads_client.get_service("AdGroupAdService")
        ad_group_ad_operation = self.google_ads_client.get_type("AdGroupAdOperation")
        ad_group_ad = ad_group_ad_operation.create
        ad_group_ad.ad_group = ad_group_resource
        ad_group_ad.status = self.google_ads_client.enums.AdGroupAdStatusEnum.ENABLED
        ad_group_ad.ad.final_urls.append(final_url)
        
        responsive_search_ad = ad_group_ad.ad.responsive_search_ad
        
        # Add headlines (min 3, max 15)
        for headline in headlines[:15]:
            headline_asset = self.google_ads_client.get_type("AdTextAsset")
            headline_asset.text = headline[:30]
            responsive_search_ad.headlines.append(headline_asset)
        
        # Add descriptions (min 2, max 4)
        for description in descriptions[:4]:
            description_asset = self.google_ads_client.get_type("AdTextAsset")
            description_asset.text = description[:90]
            responsive_search_ad.descriptions.append(description_asset)
        
        try:
            ad_group_ad_service.mutate_ad_group_ads(
                customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
                operations=[ad_group_ad_operation]
            )
            print(f"✅ Ad created successfully")
        except GoogleAdsException as ex:
            print(f"⚠️ Ad creation warning: {ex}")
            raise
            raise

# ============== ROUTES ==============
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        return RedirectResponse(url="/dashboard", status_code=303)
    else:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    # Load the comprehensive dashboard from file
    dashboard_path = os.path.join(os.path.dirname(__file__), 'templates', 'dashboard.html')
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/create-account")
async def create_account_endpoint(request: CreateAccountRequest):
    try:
        if not request.consent_acknowledged:
            raise HTTPException(status_code=400, detail="User consent is required")
        automation = GoogleAdsAutomation()
        result = automation.create_google_ads_account(
            business_name=request.business_name,
            currency_code=request.currency_code,
            time_zone=request.time_zone
        )
        return result
    except GoogleAdsException as ex:
        error_msg = "Google Ads API Error: "
        if ex.failure and ex.failure.errors:
            for error in ex.failure.errors:
                error_msg += f"{error.message}; "
        else:
            error_msg += str(ex)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/api/update-campaign-status/{campaign_id}")
async def update_campaign_status_endpoint(campaign_id: str, request: UpdateCampaignStatusRequest):
    try:
        automation = GoogleAdsAutomation()
        result = automation.update_campaign_status(
            customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
            campaign_id=campaign_id,
            new_status=request.status
        )
        return result
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

# NEW: R.27 - Campaign Editing
@app.post("/api/update-campaign/{campaign_id}")
async def update_campaign_details(campaign_id: str, request: Dict):
    try:
        automation = GoogleAdsAutomation()
        campaign_service = automation.google_ads_client.get_service("CampaignService")
        campaign_budget_service = automation.google_ads_client.get_service("CampaignBudgetService")
        
        operations = []
        
        # Update campaign name if provided
        if 'name' in request:
            campaign_operation = automation.google_ads_client.get_type("CampaignOperation")
            campaign = campaign_operation.update
            campaign.resource_name = campaign_service.campaign_path(GOOGLE_ADS_TARGET_CUSTOMER_ID, campaign_id)
            campaign.name = request['name']
            campaign_operation.update_mask.paths.append("name")
            operations.append(campaign_operation)
        
        # Update budget if provided
        if 'daily_budget' in request:
            # Get campaign budget resource name first
            ga_service = automation.google_ads_client.get_service("GoogleAdsService")
            query = f"SELECT campaign.campaign_budget FROM campaign WHERE campaign.id = {campaign_id}"
            req = automation.google_ads_client.get_type("SearchGoogleAdsRequest")
            req.customer_id = GOOGLE_ADS_TARGET_CUSTOMER_ID
            req.query = query
            response = ga_service.search(request=req)
            budget_resource_name = None
            for row in response:
                budget_resource_name = row.campaign.campaign_budget
                break
            
            if budget_resource_name:
                budget_operation = automation.google_ads_client.get_type("CampaignBudgetOperation")
                budget = budget_operation.update
                budget.resource_name = budget_resource_name
                budget.amount_micros = int(request['daily_budget'] * 1000000)
                budget_operation.update_mask.paths.append("amount_micros")
                campaign_budget_service.mutate_campaign_budgets(
                    customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
                    operations=[budget_operation]
                )
        
        if operations:
            campaign_service.mutate_campaigns(
                customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
                operations=operations
            )
        
        return {"success": True, "message": "Campaign updated"}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.get("/api/campaigns")
async def get_campaigns():
    try:
        automation = GoogleAdsAutomation()
        campaigns = automation.get_all_campaigns(GOOGLE_ADS_TARGET_CUSTOMER_ID)
        return {"success": True, "campaigns": campaigns, "total": len(campaigns)}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.get("/api/campaign-performance/{campaign_id}")
async def get_campaign_performance_endpoint(campaign_id: str, date_range: str = "LAST_30_DAYS"):
    try:
        automation = GoogleAdsAutomation()
        performance = automation.get_campaign_performance(GOOGLE_ADS_TARGET_CUSTOMER_ID, campaign_id, date_range)
        keywords = automation.get_keyword_performance(GOOGLE_ADS_TARGET_CUSTOMER_ID, campaign_id, date_range)
        ads = automation.get_ad_performance(GOOGLE_ADS_TARGET_CUSTOMER_ID, campaign_id, date_range)
        return {"success": True, "performance": performance, "keywords": keywords, "ads": ads}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

# NEW: R.28, R.29 - Ad Group Management
@app.get("/api/campaign/{campaign_id}/ad-groups")
async def get_ad_groups(campaign_id: str):
    try:
        automation = GoogleAdsAutomation()
        ga_service = automation.google_ads_client.get_service("GoogleAdsService")
        query = f"""
            SELECT
                ad_group.id,
                ad_group.name,
                ad_group.status,
                metrics.clicks,
                metrics.cost_micros,
                metrics.impressions
            FROM ad_group
            WHERE campaign.id = {campaign_id}
            AND segments.date DURING LAST_30_DAYS
        """
        request = automation.google_ads_client.get_type("SearchGoogleAdsRequest")
        request.customer_id = GOOGLE_ADS_TARGET_CUSTOMER_ID
        request.query = query
        response = ga_service.search(request=request)
        
        ad_groups_dict = {}
        for row in response:
            ag_id = str(row.ad_group.id)
            if ag_id not in ad_groups_dict:
                ad_groups_dict[ag_id] = {
                    "id": ag_id,
                    "name": row.ad_group.name,
                    "status": row.ad_group.status.name,
                    "clicks": 0,
                    "cost": 0,
                    "impressions": 0
                }
            ad_groups_dict[ag_id]["clicks"] += row.metrics.clicks
            ad_groups_dict[ag_id]["cost"] += row.metrics.cost_micros / 1000000
            ad_groups_dict[ag_id]["impressions"] += row.metrics.impressions
        
        return {"success": True, "ad_groups": list(ad_groups_dict.values())}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/api/campaign/{campaign_id}/ad-groups")
async def create_ad_group(campaign_id: str, request: Dict):
    try:
        automation = GoogleAdsAutomation()
        campaign_resource = f"customers/{GOOGLE_ADS_TARGET_CUSTOMER_ID}/campaigns/{campaign_id}"
        ad_group_resource = automation.create_ad_group(campaign_resource, request['name'], request.get('cpc_bid', 1.0))
        return {"success": True, "resource_name": ad_group_resource}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/api/ad-group/{ad_group_id}/update")
async def update_ad_group(ad_group_id: str, request: Dict):
    try:
        automation = GoogleAdsAutomation()
        ad_group_service = automation.google_ads_client.get_service("AdGroupService")
        ad_group_operation = automation.google_ads_client.get_type("AdGroupOperation")
        ad_group = ad_group_operation.update
        ad_group.resource_name = ad_group_service.ad_group_path(GOOGLE_ADS_TARGET_CUSTOMER_ID, ad_group_id)
        
        if 'name' in request:
            ad_group.name = request['name']
            ad_group_operation.update_mask.paths.append("name")
        
        if 'status' in request:
            status_enum = automation.google_ads_client.enums.AdGroupStatusEnum
            if request['status'].upper() == "PAUSED":
                ad_group.status = status_enum.PAUSED
            elif request['status'].upper() == "ENABLED":
                ad_group.status = status_enum.ENABLED
            ad_group_operation.update_mask.paths.append("status")
        
        ad_group_service.mutate_ad_groups(
            customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
            operations=[ad_group_operation]
        )
        return {"success": True}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

# NEW: R.31 - Keyword Management
@app.get("/api/campaign/{campaign_id}/keywords-full")
async def get_keywords_full(campaign_id: str):
    try:
        automation = GoogleAdsAutomation()
        ga_service = automation.google_ads_client.get_service("GoogleAdsService")
        query = f"""
            SELECT
                ad_group_criterion.criterion_id,
                ad_group_criterion.keyword.text,
                ad_group_criterion.keyword.match_type,
                ad_group_criterion.status,
                ad_group.id,
                ad_group.name,
                metrics.clicks,
                metrics.cost_micros,
                metrics.impressions
            FROM keyword_view
            WHERE campaign.id = {campaign_id}
            AND segments.date DURING LAST_30_DAYS
        """
        request = automation.google_ads_client.get_type("SearchGoogleAdsRequest")
        request.customer_id = GOOGLE_ADS_TARGET_CUSTOMER_ID
        request.query = query
        response = ga_service.search(request=request)
        
        keywords_dict = {}
        for row in response:
            kw_id = str(row.ad_group_criterion.criterion_id)
            ag_id = str(row.ad_group.id)
            key = f"{ag_id}_{kw_id}"
            if key not in keywords_dict:
                keywords_dict[key] = {
                    "id": kw_id,
                    "ad_group_id": ag_id,
                    "ad_group_name": row.ad_group.name,
                    "keyword": row.ad_group_criterion.keyword.text,
                    "match_type": row.ad_group_criterion.keyword.match_type.name,
                    "status": row.ad_group_criterion.status.name,
                    "clicks": 0,
                    "cost": 0,
                    "impressions": 0
                }
            keywords_dict[key]["clicks"] += row.metrics.clicks
            keywords_dict[key]["cost"] += row.metrics.cost_micros / 1000000
            keywords_dict[key]["impressions"] += row.metrics.impressions
        
        return {"success": True, "keywords": list(keywords_dict.values())}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/api/ad-group/{ad_group_id}/keywords")
async def add_keyword(ad_group_id: str, request: Dict):
    try:
        automation = GoogleAdsAutomation()
        ad_group_resource = f"customers/{GOOGLE_ADS_TARGET_CUSTOMER_ID}/adGroups/{ad_group_id}"
        automation.add_keywords(ad_group_resource, [request])
        return {"success": True}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/api/keyword/{ad_group_id}/{criterion_id}/update")
async def update_keyword(ad_group_id: str, criterion_id: str, request: Dict):
    try:
        automation = GoogleAdsAutomation()
        ad_group_criterion_service = automation.google_ads_client.get_service("AdGroupCriterionService")
        operation = automation.google_ads_client.get_type("AdGroupCriterionOperation")
        criterion = operation.update
        criterion.resource_name = ad_group_criterion_service.ad_group_criterion_path(
            GOOGLE_ADS_TARGET_CUSTOMER_ID, ad_group_id, criterion_id
        )
        
        status_enum = automation.google_ads_client.enums.AdGroupCriterionStatusEnum
        if request['status'].upper() == "PAUSED":
            criterion.status = status_enum.PAUSED
        elif request['status'].upper() == "ENABLED":
            criterion.status = status_enum.ENABLED
        operation.update_mask.paths.append("status")
        
        ad_group_criterion_service.mutate_ad_group_criteria(
            customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
            operations=[operation]
        )
        return {"success": True}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

# NEW: R.32 - Ad Management
@app.get("/api/campaign/{campaign_id}/ads-full")
async def get_ads_full(campaign_id: str):
    try:
        automation = GoogleAdsAutomation()
        ga_service = automation.google_ads_client.get_service("GoogleAdsService")
        query = f"""
            SELECT
                ad_group_ad.ad.id,
                ad_group_ad.ad.type,
                ad_group_ad.status,
                ad_group.id,
                ad_group.name,
                ad_group_ad.ad.responsive_search_ad.headlines,
                ad_group_ad.ad.responsive_search_ad.descriptions,
                metrics.clicks,
                metrics.cost_micros,
                metrics.impressions
            FROM ad_group_ad
            WHERE campaign.id = {campaign_id}
            AND segments.date DURING LAST_30_DAYS
        """
        request = automation.google_ads_client.get_type("SearchGoogleAdsRequest")
        request.customer_id = GOOGLE_ADS_TARGET_CUSTOMER_ID
        request.query = query
        response = ga_service.search(request=request)
        
        ads_dict = {}
        for row in response:
            ad_id = str(row.ad_group_ad.ad.id)
            ag_id = str(row.ad_group.id)
            key = f"{ag_id}_{ad_id}"
            if key not in ads_dict:
                headlines = [h.text for h in row.ad_group_ad.ad.responsive_search_ad.headlines[:3]]
                ads_dict[key] = {
                    "id": ad_id,
                    "ad_group_id": ag_id,
                    "ad_group_name": row.ad_group.name,
                    "type": row.ad_group_ad.ad.type_.name,
                    "status": row.ad_group_ad.status.name,
                    "preview": " | ".join(headlines) if headlines else "N/A",
                    "clicks": 0,
                    "cost": 0,
                    "impressions": 0
                }
            ads_dict[key]["clicks"] += row.metrics.clicks
            ads_dict[key]["cost"] += row.metrics.cost_micros / 1000000
            ads_dict[key]["impressions"] += row.metrics.impressions
        
        return {"success": True, "ads": list(ads_dict.values())}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/api/ad-group/{ad_group_id}/ads")
async def create_ad_endpoint(ad_group_id: str, request: Dict):
    try:
        automation = GoogleAdsAutomation()
        ad_group_resource = f"customers/{GOOGLE_ADS_TARGET_CUSTOMER_ID}/adGroups/{ad_group_id}"
        automation.create_ad(
            ad_group_resource,
            request['headlines'],
            request['descriptions'],
            request['final_url']
        )
        return {"success": True}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/api/ad/{ad_group_id}/{ad_id}/update")
async def update_ad(ad_group_id: str, ad_id: str, request: Dict):
    try:
        automation = GoogleAdsAutomation()
        ad_group_ad_service = automation.google_ads_client.get_service("AdGroupAdService")
        operation = automation.google_ads_client.get_type("AdGroupAdOperation")
        ad_group_ad = operation.update
        ad_group_ad.resource_name = ad_group_ad_service.ad_group_ad_path(
            GOOGLE_ADS_TARGET_CUSTOMER_ID, ad_group_id, ad_id
        )
        
        status_enum = automation.google_ads_client.enums.AdGroupAdStatusEnum
        if request['status'].upper() == "PAUSED":
            ad_group_ad.status = status_enum.PAUSED
        elif request['status'].upper() == "ENABLED":
            ad_group_ad.status = status_enum.ENABLED
        operation.update_mask.paths.append("status")
        
        ad_group_ad_service.mutate_ad_group_ads(
            customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
            operations=[operation]
        )
        return {"success": True}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

# NEW: R.33 - Negative Keyword Management
@app.get("/api/campaign/{campaign_id}/negative-keywords")
async def get_negative_keywords(campaign_id: str):
    try:
        automation = GoogleAdsAutomation()
        ga_service = automation.google_ads_client.get_service("GoogleAdsService")
        query = f"""
            SELECT
                campaign_criterion.criterion_id,
                campaign_criterion.keyword.text,
                campaign_criterion.keyword.match_type,
                campaign_criterion.negative
            FROM campaign_criterion
            WHERE campaign.id = {campaign_id}
            AND campaign_criterion.type = KEYWORD
            AND campaign_criterion.negative = TRUE
        """
        request = automation.google_ads_client.get_type("SearchGoogleAdsRequest")
        request.customer_id = GOOGLE_ADS_TARGET_CUSTOMER_ID
        request.query = query
        response = ga_service.search(request=request)
        
        negatives = []
        for row in response:
            negatives.append({
                "id": str(row.campaign_criterion.criterion_id),
                "keyword": row.campaign_criterion.keyword.text,
                "match_type": row.campaign_criterion.keyword.match_type.name
            })
        
        return {"success": True, "negative_keywords": negatives}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/api/campaign/{campaign_id}/negative-keywords")
async def add_negative_keyword(campaign_id: str, request: Dict):
    try:
        automation = GoogleAdsAutomation()
        campaign_criterion_service = automation.google_ads_client.get_service("CampaignCriterionService")
        operation = automation.google_ads_client.get_type("CampaignCriterionOperation")
        criterion = operation.create
        criterion.campaign = f"customers/{GOOGLE_ADS_TARGET_CUSTOMER_ID}/campaigns/{campaign_id}"
        criterion.keyword.text = request['keyword']
        
        match_type_map = {
            "BROAD": automation.google_ads_client.enums.KeywordMatchTypeEnum.BROAD,
            "PHRASE": automation.google_ads_client.enums.KeywordMatchTypeEnum.PHRASE,
            "EXACT": automation.google_ads_client.enums.KeywordMatchTypeEnum.EXACT
        }
        criterion.keyword.match_type = match_type_map.get(request.get('match_type', 'BROAD'))
        criterion.negative = True
        
        campaign_criterion_service.mutate_campaign_criteria(
            customer_id=GOOGLE_ADS_TARGET_CUSTOMER_ID,
            operations=[operation]
        )
        return {"success": True}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/api/create-smart-campaign")
async def create_smart_campaign(request: SimpleCampaignRequest):
    try:
        automation = GoogleAdsAutomation()
        ai_data = automation.generate_keywords_with_ai(request.product_url, request.objective, request.product_description)
        campaign_resource, campaign_name, optimal_budget, campaign_id = automation.create_smart_campaign(request)
        automation.add_smart_geo_targeting(campaign_resource, request.product_url)
        automation.add_smart_language_targeting(campaign_resource)
        ad_group_resource = automation.create_ad_group(campaign_resource, f"{campaign_name}_AdGroup", cpc_bid=2.0)
        automation.add_keywords(ad_group_resource, ai_data['keywords'])
        automation.create_ad(ad_group_resource, ai_data['headlines'], ai_data['descriptions'], request.product_url)
        return {
            "success": True,
            "campaign_name": campaign_name,
            "campaign_id": campaign_id,
            "daily_budget": optimal_budget,
            "keywords_count": len(ai_data['keywords'])
        }
    except GoogleAdsException as ex:
        error_msg = "Google Ads API Error: "
        if ex.failure and ex.failure.errors:
            for error in ex.failure.errors:
                error_msg += f"{error.message}; "
        else:
            error_msg += str(ex)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "10.0 - Full RMF Compliant (R.27-R.33)"}

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    print("=" * 70)
    print("🚀 Google Ads AI Automation - FULL RMF DEMO")
    print(f"🎯 TARGETING TEST ACCOUNT: {GOOGLE_ADS_TARGET_CUSTOMER_ID}")
    print("=" * 70)
    print("📊 FULL RMF FEATURES IMPLEMENTED:")
    print(" ✅ R.25 - Pause/Enable Campaigns")
    print(" ✅ R.26 - Campaign List View")
    print(" ✅ R.27 - Campaign Editing (Name & Budget)")
    print(" ✅ R.28 - Ad Group Create/Edit")
    print(" ✅ R.29 - Ad Group Performance View")
    print(" ✅ R.30 - Keyword & Ad Detail View")
    print(" ✅ R.31 - Keyword Create/Edit")
    print(" ✅ R.32 - Ad Create/Edit")
    print(" ✅ R.33 - Negative Keywords")
    print(" ✅ R.1, R.2, R.3, R.4, R.5 - Core Metrics")
    print("=" * 70)
    print("📍 Run this and go to http://127.0.0.1:8000/")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)