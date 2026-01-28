import numpy as np
from typing import Dict, List
import random
from collections import defaultdict

class RetrievalEvaluator:
    def __init__(self, metrics=None, k_values=None):
        """
        :param metrics: 指标列表，例如 ["precision", "recall", "map", "mrr", "ndcg"]
        :param k_values: 针对需要@k的指标，指定k的取值，例如 [1, 5, 10]
        """
        self.metrics = metrics or ["precision", "recall", "map", "mrr", "ndcg"]
        self.k_values = k_values or [1, 5, 10]

    def precision_at_k(self, retrieve_list, gold_list, k):
        retrieved_k = retrieve_list[:k]
        return len(set(retrieved_k) & set(gold_list)) / k

    def recall_at_k(self, retrieve_list, gold_list, k):
        retrieved_k = retrieve_list[:k]
        return len(set(retrieved_k) & set(gold_list)) / len(gold_list) if gold_list else 0

    def average_precision(self, retrieve_list, gold_list):
        hits, sum_prec = 0, 0.0
        for i, doc in enumerate(retrieve_list, start=1):
            if doc in gold_list:
                hits += 1
                sum_prec += hits / i
        return sum_prec / len(gold_list) if gold_list else 0

    def mrr(self, retrieve_list, gold_list):
        for i, doc in enumerate(retrieve_list, start=1):
            if doc in gold_list:
                return 1 / i
        return 0

    def ndcg_at_k(self, retrieve_list, gold_list, k):
        rels = [1 if doc in gold_list else 0 for doc in retrieve_list[:k]]
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rels))
        ideal_rels = sorted(rels, reverse=True)
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
        return dcg / idcg if idcg > 0 else 0

    def evaluate_query(self, retrieve_list, gold_list):
        """
        针对单个 query 的评估
        :return: dict, 包含各类指标
        """
        results = {}

        for metric in self.metrics:
            if metric in ["precision", "recall", "ndcg"]:
                for k in self.k_values:
                    if metric == "precision":
                        results[f"P@{k}"] = self.precision_at_k(retrieve_list, gold_list, k)
                    elif metric == "recall":
                        results[f"R@{k}"] = self.recall_at_k(retrieve_list, gold_list, k)
                    elif metric == "ndcg":
                        results[f"NDCG@{k}"] = self.ndcg_at_k(retrieve_list, gold_list, k)

            elif metric == "map":
                results["MAP"] = self.average_precision(retrieve_list, gold_list)

            elif metric == "mrr":
                results["MRR"] = self.mrr(retrieve_list, gold_list)

        return results

    def evaluate_all(self, queries):
        """
        针对多个 query 的整体评估
        :param queries: list of dict, 每个元素为 {"query_id": str, "gold": [...], "retrieve": [...]}
        :return: dict, 每个指标的平均值
        """
        all_results = []
        for q in queries:
            res = self.evaluate_query(q["retrieve"], q["gold"])
            all_results.append(res)

        # 聚合平均
        aggregated = {}
        for key in all_results[0].keys():
            aggregated[key] = np.mean([r[key] for r in all_results])

        return aggregated

    def evaluate_custom_queries(self, queries, retriever_fn) -> Dict[int, Dict]:
        """
        for custom query list
        :param queries: list of dict, 每个元素为 {"query_id": str, "query": str, "gold": list}
        :param retriever_fn: callable, args: (queries, top_k) -> list of retrieval results
        :return: dict, {top_k: {metric: value, by_type: {...}}}
        """
        top_ks = self.k_values
        max_k = max(top_ks)

        all_results = retriever_fn(queries, max_k)

        eval_inputs = []
        for q, res in zip(queries, all_results):
            eval_inputs.append({
                **q,
                **res,
                "retrieve": res["associated_chunks"][:max_k]
            })

        # 整体聚合评估
        aggregated_results = self.evaluate_all(eval_inputs)

        results_by_k = {}
        for top_k in top_ks:
            # 整体指标
            results_by_k[top_k] = {
                "P": aggregated_results.get(f"P@{top_k}", 0),
                "R": aggregated_results.get(f"R@{top_k}", 0),
                "NDCG": aggregated_results.get(f"NDCG@{top_k}", 0),
                "MAP": aggregated_results.get("MAP", 0),
                "MRR": aggregated_results.get("MRR", 0),
            }

        return results_by_k

    
    def evaluate_dataset(self, dataset,valid_entries,retriever_fn,) -> Dict[int, Dict]:
        # 初始化评估器，包含所有需要的指标
        '''
        dataset: Openwiki/Chartmrag
        valid_entries: list of int, valid entries's index
        retriever_fn : calllable, args: all_queries, top_k
        top_ks: list of int, top_k
        return: dict, {top_k: {metric: value}}
        '''
        top_ks = self.k_values
        eval_queries = []
        valid_qa_ids = set()
        for idx in valid_entries: 
            entry = dataset[idx]
            valid_qa_ids.update(entry.qa_ids)

        for qa_id in valid_qa_ids:
            qa_info = dataset.get_qa(qa_id)
            
            #if "chart" in qa_id:
            eval_queries.append({
                "query_id": qa_id,
                "gold": qa_info["gt_ids"],  # 转换为列表
                "query": qa_info["query"]
            })
        
        # 批量检索结果（使用最大top_k提高效率）
        max_k = max(top_ks)
        # Adjust 把query的相关信息也传进去
        #all_queries = [q["query"] for q in eval_queries]
        all_results = retriever_fn(eval_queries, max_k)

        eval_inputs = []
        for q, res in zip(eval_queries, all_results):
            input = {
                **q,
                **res,
                "query_type":q["query_id"].rsplit("_", 1)[0],
                "retrieve": res["associated_chunks"][:max_k]
                    # 确保不超过max_k
            }
            eval_inputs.append(input)
           #print(input)
        
        random_eval_inputs = random.sample(eval_inputs, k=min(10, len(eval_inputs)))
        print("random eval inputs:")
        for input in random_eval_inputs:
            simple_eval_print(input)
        aggregated_results = self.evaluate_all(eval_inputs)
        
        results_by_k = {}
        for top_k in top_ks:
            # 整体指标
            results_by_k[top_k] = {
                "P": aggregated_results.get(f"P@{top_k}", 0),
                "R": aggregated_results.get(f"R@{top_k}", 0),
                "NDCG": aggregated_results.get(f"NDCG@{top_k}", 0),
                "MAP": aggregated_results.get("MAP", 0),
                "MRR": aggregated_results.get("MRR", 0),
                "by_type": {}
            }

            # 按类型分组统计
            type_groups = defaultdict(list)
            for item in eval_inputs:
                type_groups[item["query_type"]].append(item)

            for qtype, items in type_groups.items():
                # 对每个类型单独调用 evaluate_all 计算指标
                type_results = self.evaluate_all([
                    {
                        "query_id": it["query_id"],
                        "gold": it["gold"],
                        "retrieve": it["retrieve"][:top_k]
                    }
                    for it in items
                ])
                results_by_k[top_k]["by_type"][qtype] = {
                    "P": type_results.get(f"P@{top_k}", 0),
                    "R": type_results.get(f"R@{top_k}", 0),
                    "NDCG": type_results.get(f"NDCG@{top_k}", 0)
                }
        return results_by_k

def print_evaluation_results(results_by_k: Dict[int, Dict], by_type=False):
    """打印评估结果，包含按类型的统计"""
    print("\nEvaluation Results:")
    for top_k in sorted(results_by_k.keys()):
        metrics = results_by_k[top_k]
        print(f"\n=== Top-{top_k} Overall ===")
        print(f"P={metrics['P']:.4f}, R={metrics['R']:.4f}, "
              f"NDCG={metrics['NDCG']:.4f}, MAP={metrics['MAP']:.4f}, MRR={metrics['MRR']:.4f}")
        
        if by_type:
            print("\n--- By Query Type ---")
            print("Type\tPrecision\tRecall\tNDCG")
            for qtype, vals in metrics["by_type"].items():
                print(f"{qtype}\t{vals['P']:.4f}\t{vals['R']:.4f}\t{vals['NDCG']:.4f}")


def simple_eval_print(eval_input):
    query_id = eval_input["query_id"]
    gold = eval_input["gold"]
    query = eval_input["query"]
    gt_ents = eval_input["gt_entities"]
    gt_ents = [ent.summary() for ent in gt_ents]
    similar_entities = eval_input["similar_entities"]
    similar_entities = [(ent.summary(), score) for ent, score in similar_entities]
    associated_chunks = eval_input["associated_chunks"]
    print({
        "query_id": query_id,
        "query": query,
        "gold": gold,
        "gt_entities": gt_ents,
        "similar_entities": similar_entities,
        "associated_chunks": associated_chunks
    })
