#!/usr/bin/env python3
"""
Step 02: Enhanced Web Data Collection from ClinicalTrials.gov
"""

import requests
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm

class ClinicalTrialsWebCollector:
    def __init__(self, output_dir="data"):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.rate_limit = 1.2  # seconds between requests
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def search_trials(self, query_term="", max_studies=500):
        all_studies, page_token, page_size = [], None, 1000
        pbar = tqdm(total=max_studies, desc=f"Fetching {query_term}")
        while len(all_studies) < max_studies:
            try:
                params = {
                    "format": "json",
                    "pageSize": min(page_size, max_studies - len(all_studies))
                }
                if query_term: params["query.cond"] = query_term
                if page_token: params["pageToken"] = page_token
                r = requests.get(self.base_url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                if 'studies' not in data or not data['studies']:
                    break
                processed = [self._extract_fields(study) for study in data['studies']]
                all_studies.extend(filter(None, processed))
                pbar.update(len(processed))
                page_token = data.get('nextPageToken')
                if not page_token: break
                time.sleep(self.rate_limit)
            except Exception as e:
                print(f"⚠️ Error: {e}")
                break
        pbar.close()
        df = pd.DataFrame(all_studies[:max_studies])
        if not df.empty:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out_file = self.output_dir / f"raw_clinical_trials_{timestamp}.csv"
            df.to_csv(out_file, index=False)
            print(f"✅ {len(df)} trials saved to {out_file}")
        else:
            print("❌ No studies collected")
        return df

    def _extract_fields(self, study):
        try:
            protocol = study.get('protocolSection', {})
            identification = protocol.get('identificationModule', {})
            description = protocol.get('descriptionModule', {})
            status = protocol.get('statusModule', {})
            design = protocol.get('designModule', {})
            design_info = design.get('designInfo', {})
            masking_info = design_info.get('maskingInfo', {})
            enrollment_info = design.get('enrollmentInfo', {})

            return {
                "nct_id": identification.get("nctId", ""),
                "brief_title": identification.get("briefTitle", ""),
                "official_title": identification.get("officialTitle", ""),
                "brief_summary": description.get("briefSummary", ""),
                "detailed_description": description.get("detailedDescription", ""),
                "overall_status": status.get("overallStatus", ""),
                "why_stopped": status.get("whyStopped", ""),
                "start_date": status.get("startDateStruct", {}).get("date", ""),
                "completion_date": status.get("completionDateStruct", {}).get("date", ""),
                "primary_completion_date": status.get("primaryCompletionDateStruct", {}).get("date", ""),
                "study_type": design.get("studyType", ""),
                "phases": '|'.join(design.get("phases", [])) if design.get("phases") else "",
                "allocation": design_info.get("allocation", ""),
                "intervention_model": design_info.get("interventionModel", ""),
                "masking": masking_info.get("masking", ""),
                "primary_purpose": design_info.get("primaryPurpose", ""),
                "enrollment_count": enrollment_info.get("count", 0),
                "enrollment_type": enrollment_info.get("type", "")
            }
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return None

if __name__ == "__main__":
    collector = ClinicalTrialsWebCollector("data")
    # Example: download 300 cancer, diabetes and hypertension trials
    _ = collector.search_trials("cancer", 500)
    _ = collector.search_trials("diabetes", 500)
    _ = collector.search_trials("hypertension", 500)
    print("🚀 Data collection completed.")
