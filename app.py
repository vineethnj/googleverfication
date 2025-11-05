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

app = FastAPI(title="Google Ads AI Automation - RMF Demo")
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

# Initialize Database (assuming Database is defined elsewhere; if not, remove or implement)
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
    status: str  # Should be "ENABLED" or "PAUSED"

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
        ad_group_ad_service = self.google_ads_client.get_service("AdGroupAdService")
        ad_group_ad_operation = self.google_ads_client.get_type("AdGroupAdOperation")
        ad_group_ad = ad_group_ad_operation.create
        ad_group_ad.ad_group = ad_group_resource
        ad_group_ad.status = self.google_ads_client.enums.AdGroupAdStatusEnum.ENABLED
        ad_group_ad.ad.final_urls.append(final_url)
        responsive_search_ad = ad_group_ad.ad.responsive_search_ad
        for headline in headlines[:15]:
            headline_asset = self.google_ads_client.get_type("AdTextAsset")
            headline_asset.text = headline[:30]
            responsive_search_ad.headlines.append(headline_asset)
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
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>RMF Demo Dashboard - Growthsynth</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f0f2f5; color: #1c1e21; margin: 0; padding: 20px; }
            .container { max-width: 1400px; margin: auto; }
            .header { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); margin-bottom: 20px; }
            h1 { font-size: 28px; } h2 { font-size: 22px; margin-bottom: 15px; } h3 { font-size: 18px; margin-bottom: 10px; }
            .tabs { display: flex; border-bottom: 1px solid #ddd; margin-bottom: 20px; }
            .tab { padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; font-weight: 600; color: #606770; }
            .tab.active { color: #1877f2; border-bottom: 3px solid #1877f2; }
            .tab-content { display: none; } .tab-content.active { display: block; }
            .card { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); margin-bottom: 20px; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f0f2f5; font-weight: 600; }
            .status-enabled { color: #38a169; font-weight: bold; } .status-paused { color: #e53e3e; font-weight: bold; }
            .loading, .error { text-align: center; padding: 40px; font-size: 16px; color: #606770; }
            .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #1877f2; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .switch { position: relative; display: inline-block; width: 50px; height: 28px; }
            .switch input { opacity: 0; width: 0; height: 0; }
            .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 28px; }
            .slider:before { position: absolute; content: ""; height: 20px; width: 20px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }
            input:checked + .slider { background-color: #2196F3; }
            input:checked + .slider:before { transform: translateX(22px); }
            #details-view { border: 1px solid #ddd; margin-top: 20px; padding: 20px; border-radius: 8px; }
            .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .metric-card { background: #f0f2f5; padding: 15px; border-radius: 8px; text-align: center; }
            .metric-card-label { font-size: 14px; color: #606770; margin-bottom: 5px; }
            .metric-card-value { font-size: 24px; font-weight: bold; color: #1c1e21; }
            .date-picker { margin-bottom: 20px; }
            .date-picker select { padding: 8px; border-radius: 6px; border: 1px solid #ccc; font-size: 14px; }
            .btn { padding: 8px 16px; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; }
            .btn-details { background-color: #e4e6eb; color: #050505; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>Growthsynth RMF Demo</h1></div>
            <div class="tabs">
                <button class="tab" onclick="showTab('campaigns')">📋 All Campaigns</button>
                <button class="tab" onclick="showTab('create-campaign')">🚀 Create Campaign</button>
                <button class="tab" onclick="showTab('create-account')">➕ Create Account</button>
            </div>
            <div id="campaigns-tab" class="tab-content">
                <div class="card">
                    <h2>Campaign Overview</h2>
                    <div class="date-picker">
                        <label for="date-range-select" style="margin-right: 10px; font-weight: bold;">Date Range:</label>
                        <select id="date-range-select">
                            <option value="LAST_7_DAYS">Last 7 Days</option>
                            <option value="LAST_14_DAYS">Last 14 Days</option>
                            <option value="LAST_30_DAYS" selected>Last 30 Days</option>
                            <option value="THIS_MONTH">This Month</option>
                            <option value="LAST_MONTH">Last Month</option>
                        </select>
                    </div>
                    <div id="campaigns-list-container">
                        <div class="loading"><div class="spinner"></div>Loading Campaigns...</div>
                    </div>
                    <div id="details-view" style="display:none;"></div>
                </div>
            </div>
            <div id="create-campaign-tab" class="tab-content"><div class="card"><h2>Create Campaign</h2><p>Functionality as shown in design document.</p></div></div>
            <div id="create-account-tab" class="tab-content"><div class="card"><h2>Create Account</h2><p>Functionality as shown in design document.</p></div></div>
        </div>
        <script>
            let campaignsData = [];
            function showTab(tabId) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
                document.getElementById(tabId + '-tab').classList.add('active');
                document.querySelectorAll('.tab').forEach(el => {
                    if (el.getAttribute('onclick') === "showTab('" + tabId + "')") {
                        el.classList.add('active');
                    }
                });
                if (tabId === 'campaigns') {
                    loadCampaigns();
                }
            }

            async function loadCampaigns() {
                const container = document.getElementById('campaigns-list-container');
                container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading Campaigns...</div>';
                document.getElementById('details-view').style.display = 'none';
                try {
                    const response = await fetch('/api/campaigns');
                    if (!response.ok) throw new Error('Failed to fetch campaigns');
                    const data = await response.json();
                    campaignsData = data.campaigns;
                    renderCampaignsTable(campaignsData);
                } catch (error) {
                    container.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            }

            function renderCampaignsTable(campaigns) {
                const container = document.getElementById('campaigns-list-container');
                if (campaigns.length === 0) {
                    container.innerHTML = '<div class="loading">No campaigns found in this test account. Please create one.</div>';
                    return;
                }
                let tableHTML = `
                    <table>
                        <thead>
                            <tr>
                                <th>Campaign Name</th>
                                <th>Status</th>
                                <th>Budget/Day</th>
                                <th>Enable/Pause (R.25)</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>`;
                campaigns.forEach(c => {
                    tableHTML += `
                        <tr>
                            <td>${c.name}</td>
                            <td><span class="status-${c.status.toLowerCase()}">${c.status}</span></td>
                            <td>$${c.budget.toFixed(2)}</td>
                            <td>
                                <label class="switch">
                                    <input type="checkbox" ${c.status === 'ENABLED' ? 'checked' : ''} onchange="toggleCampaignStatus('${c.id}', this.checked)">
                                    <span class="slider"></span>
                                </label>
                            </td>
                            <td><button class="btn btn-details" onclick="viewDetails('${c.id}')">View Details</button></td>
                        </tr>`;
                });
                tableHTML += `</tbody></table>`;
                container.innerHTML = tableHTML;
            }

            async function toggleCampaignStatus(campaignId, isEnabled) {
                const newStatus = isEnabled ? 'ENABLED' : 'PAUSED';
                try {
                    const response = await fetch('/api/update-campaign-status/' + campaignId, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ status: newStatus })
                    });
                    if (!response.ok) throw new Error('Failed to update status');
                    loadCampaigns();
                } catch (error) {
                    alert('Error updating campaign status: ' + error.message);
                }
            }

            async function viewDetails(campaignId) {
                const detailsView = document.getElementById('details-view');
                detailsView.style.display = 'block';
                detailsView.innerHTML = '<div class="loading"><div class="spinner"></div>Loading Details...</div>';
                detailsView.dataset.campaignId = campaignId;
                const dateRange = document.getElementById('date-range-select').value;
                try {
                    const response = await fetch(`/api/campaign-performance/${campaignId}?date_range=${dateRange}`);
                    if (!response.ok) throw new Error('Failed to fetch performance data');
                    const data = await response.json();
                    let detailsHTML = `
                        <h3>Performance for: ${data.performance.campaign_name} (R.26)</h3>
                        <p>Date Range: ${data.performance.date_range} (R.5)</p>
                        <div class="metric-grid">
                            <div class="metric-card"><div class="metric-card-label">Impressions (R.3)</div><div class="metric-card-value">${data.performance.impressions.toLocaleString()}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Clicks (R.1)</div><div class="metric-card-value">${data.performance.clicks.toLocaleString()}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Cost (R.2)</div><div class="metric-card-value">$${data.performance.cost.toFixed(2)}</div></div>
                            <div class="metric-card"><div class="metric-card-label">Conversions (R.4)</div><div class="metric-card-value">${data.performance.conversions.toLocaleString()}</div></div>
                        </div>
                        <h3>Keywords (R.30)</h3>
                        <table>
                            <thead><tr><th>Keyword</th><th>Match Type</th><th>Status</th><th>Clicks</th><th>Cost</th></tr></thead>
                            <tbody>${data.keywords.map(k => `<tr><td>${k.keyword}</td><td>${k.match_type}</td><td>${k.status}</td><td>${k.clicks}</td><td>$${k.cost.toFixed(2)}</td></tr>`).join('')}</tbody>
                        </table>
                        <h3 style="margin-top:20px;">Ads (R.30)</h3>
                        <table>
                            <thead><tr><th>Ad ID</th><th>Ad Type</th><th>Status</th><th>Clicks</th><th>Cost</th></tr></thead>
                            <tbody>${data.ads.map(a => `<tr><td>${a.ad_id}</td><td>${a.ad_type}</td><td>${a.status}</td><td>${a.clicks}</td><td>$${a.cost.toFixed(2)}</td></tr>`).join('')}</tbody>
                        </table>
                    `;
                    detailsView.innerHTML = detailsHTML;
                } catch (error) {
                    detailsView.innerHTML = `<div class="error">Error loading details: ${error.message}</div>`;
                }
            }

            document.getElementById('date-range-select').addEventListener('change', () => {
                const detailsView = document.getElementById('details-view');
                if (detailsView.style.display === 'block' && detailsView.dataset.campaignId) {
                    viewDetails(detailsView.dataset.campaignId);
                }
            });
            showTab('campaigns');
        </script>
    </body>
    </html>
    """
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
    return {"status": "healthy", "version": "9.0 - RMF Compliant Demo"}

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    with open("templates/login.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Ads AI Automation - Login</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f0f2f5;
            color: #1c1e21;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .login-container {
            background: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            width: 350px;
            text-align: center;
        }
        h1 {
            font-size: 24px;
            margin-bottom: 20px;
            color: #1877f2;
        }
        .form-group {
            margin-bottom: 20px;
            text-align: left;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
        }
        input[type="text"],
        input[type="password"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
            box-sizing: border-box;
        }
        button[type="submit"] {
            width: 100%;
            padding: 12px;
            background-color: #1877f2;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button[type="submit"]:hover {
            background-color: #166fe5;
        }
        .error {
            color: #e53e3e;
            margin-top: 10px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>Google Ads AI Automation - Login</h1>
        <form action="/login" method="post">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" placeholder="google-review@growthsynth.com" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" placeholder="DemoAccess2024!" required>
            </div>
            {% if error %}
            <div class="error">{{ error }}</div>
            {% endif %}
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>
        """)
    print("=" * 70)
    print("🚀 Google Ads AI Automation - RMF DEMO READY")
    print(f"🎯 TARGETING TEST ACCOUNT: {GOOGLE_ADS_TARGET_CUSTOMER_ID}")
    print("=" * 70)
    print("📊 RMF FEATURES IMPLEMENTED:")
    print(" ✅ R.25 - Pause/Enable Campaigns")
    print(" ✅ R.5 - Dynamic Date Range Reporting")
    print(" ✅ R.26 - Campaign List View")
    print(" ✅ R.30 - Keyword & Ad Detail View")
    print(" ✅ R.1, R.2, R.3, R.4 - Core Metrics Reporting")
    print("=" * 70)
    print("📍 Run this and go to http://127.0.0.1:8000/")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
