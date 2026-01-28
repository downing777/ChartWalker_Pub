import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import load_from_disk, Dataset
import os


@dataclass
class ChartEntry:
    ori_id: str  
    context: List[str]
    image_paths: List[str]
    qa_ids: List[str]  


class ChartMRag(Dataset):
    """
    Dataset constructed based on ChartMRag.
    Each page is aggregated into a ChartEntry containing:
      - all charts in that page
      - all texts in that page
      - query ids related to that page
    A separate qa_index stores full QA information.
    """

    def __init__(self, config, split="train"):
        self.config = config
        self.name = "chartmrag"
        self.dataset = load_from_disk(self.config["qa_dataset"])
        chart_path = os.path.join(self.config["qa_dataset"], "chart_corpus_with_text.jsonl")
        text_path = os.path.join(self.config["qa_dataset"], "text_corpus.jsonl")

        # --- load chart corpus ---
        self.chart_corpus = []
        self.chart_map = {}
        self.chart_text_map = {}
        with open(chart_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.chart_corpus.append(data)
                self.chart_map[data["id"]] = data["chart"]
                self.chart_text_map[data["id"]] = data["text"]

        # --- load text corpus ---
        self.text_corpus = [] # [{"id":, "text":}]
        self.text_map = {}
        self.merged_text_map = {} # choose to use the merged text paragraph merged_id:text
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.text_corpus.append(data)
                self.text_map[data["id"]] = data["text"]
        self.merge_paragraph()

        # build index
        self.chart_index = self._build_chart_index() #{page_id:[chart_id]}
        self.text_index = self._build_text_index()

        # build QA index
        self.qa_index, self.page2qa_ids = self._build_qa_index()

        # aggregate into entries
        self.entries = self._build_entries()
        self.statistics()

    
    def merge_paragraph(self):
        '''
        merge the text date with in the same paragraph, in case of overly fragmented text
        '''
        paragraph_groups = defaultdict(list)
        for text_id in self.text_map:
            if text_id.startswith("paragraph_"):
                group_id = "_".join(text_id.split("_")[:3])
                paragraph_groups[group_id].append(text_id)

        for group_id, text_ids in paragraph_groups.items():
            text_ids.sort(key=lambda x: int(x.split("_")[-1]))
            
            merged_text = "\n\n".join(self.text_map[text_id] for text_id in text_ids)
            
            merged_id = group_id
            self.merged_text_map[merged_id] = merged_text
            
        pass 

    def _page_id_from_chart(self, chart_id: str) -> str:
        return chart_id.split("_")[1]

    def _page_id_from_text(self, text_id: str) -> str:
        return text_id.split("_")[1]

    def _build_chart_index(self):
        page2charts = defaultdict(list)
        for item in self.chart_corpus:
            page_id = self._page_id_from_chart(item["id"])
            page2charts[page_id].append(item["id"])
        return page2charts

    def _build_text_index(self):
        page2texts = defaultdict(list)
        for k, v in self.merged_text_map.items():
            page_id = self._page_id_from_text(k)
            page2texts[page_id].append(k)
        return page2texts

    def _build_qa_index(self):
        qa_index = {}
        page2qa_ids = defaultdict(set)

        for sample in self.dataset:
            qa_id = sample["id"]
            gt_keypoints = json.loads(sample["gt_keypoints"])
        
            gt_ids = set()
            for ref_id, keypoint in gt_keypoints.items():
                if ref_id.startswith("paragraph_"):
                    new_ref_id = "_".join(ref_id.split("_")[:3])  # 
                else:
                    new_ref_id = ref_id
                gt_ids.add(new_ref_id)
            qa_index[qa_id] = {
                "query": sample["query"],
                "answer": sample["gt_answer"],
                "gt_chart": sample["gt_chart"],
                "gt_text": sample["gt_text"],
                "gt_keypoints": gt_keypoints,
                "gt_ids": list(gt_ids)
            }

            for ref_id in qa_index[qa_id]["gt_keypoints"].keys():
                if ref_id.startswith("chart_"):
                    page_id = self._page_id_from_chart(ref_id)
                elif ref_id.startswith("paragraph_"):
                    page_id = self._page_id_from_text(ref_id)
                else:
                    continue
                page2qa_ids[page_id].add(qa_id)

        return qa_index, page2qa_ids

    def _build_entries(self):
        entries = []
        all_pages = set(self.chart_index.keys()) | set(self.text_index.keys())
        for page_id in sorted(all_pages):
            entry = ChartEntry(
                ori_id=page_id,
                context=self.text_index.get(page_id, []), # text ids
                image_paths=self.chart_index.get(page_id, []),
                qa_ids=self.page2qa_ids.get(page_id, [])
            )
            entries.append(entry)
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> ChartEntry:
        return self.entries[idx]

    def get_qa(self, qa_id: str) -> Dict[str, Any]:
        """Retrieve full QA info by qa_id."""
        return self.qa_index[qa_id]

    def statistics(self, verbose=True):
        stats = {
            "num_entries": len(self.entries),
            "avg_context_len": sum(len(e.context) for e in self.entries) / len(self.entries),
            "avg_num_charts": sum(len(e.image_paths) for e in self.entries) / len(self.entries),
            "avg_num_qas": sum(len(e.qa_ids) for e in self.entries) / len(self.entries),
            "num_qas": len(self.qa_index),
        }
        if verbose:
            print("Dataset Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
        return stats
    
