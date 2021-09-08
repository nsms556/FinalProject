# Visualization library
# from tqdm import tqdm

# CLI library
import fire

# Data library
import numpy as np
import torch

# Utils
from Utils.file import remove_file, load_json, write_json
from Utils.preprocessing import binary_songs2ids, binary_tags2ids


class ArenaEvaluator:
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, gt, rec):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)
        if len(gt)>100:
            gt = gt[:100]
        return dcg / self._idcgs[len(gt)]

    def _eval(self, gt_fname, rec_fname):
        gt_playlists = load_json(gt_fname)
        gt_dict = {g["id"]: g for g in gt_playlists}
        rec_playlists = load_json(rec_fname)
        gt_ids = set([g["id"] for g in gt_playlists])
        rec_ids = set([r["id"] for r in rec_playlists])
        if gt_ids != rec_ids:
            raise Exception("결과의 플레이리스트 수가 올바르지 않습니다.")

        rec_song_counts = [len(p["songs"]) for p in rec_playlists]
        rec_tag_counts = [len(p["tags"]) for p in rec_playlists]
        if set(rec_song_counts) != set([100]):
            raise Exception("추천 곡 결과의 개수가 맞지 않습니다.")

        if set(rec_tag_counts) != set([10]):
            raise Exception("추천 태그 결과의 개수가 맞지 않습니다.")

        rec_unique_song_counts = [len(set(p["songs"])) for p in rec_playlists]
        rec_unique_tag_counts = [len(set(p["tags"])) for p in rec_playlists]

        if set(rec_unique_song_counts) != set([100]):
            raise Exception("한 플레이리스트에 중복된 곡 추천은 허용되지 않습니다.")

        if set(rec_unique_tag_counts) != set([10]):
            raise Exception("한 플레이리스트에 중복된 태그 추천은 허용되지 않습니다.")

        music_ndcg = 0.0
        tag_ndcg = 0.0

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate_with_save(self, gt_fname, rec_fname, model_file_path, default_file_path):
        # try:
        music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
        with open(f'{default_file_path}/results.txt','a') as f:
            f.write(model_file_path)
            f.write(f"\nMusic nDCG: {music_ndcg:.6}\n")
            f.write(f"Tag nDCG: {tag_ndcg:.6}\n")
            f.write(f"Score: {score:.6}\n\n")
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Score: {score:.6}")
        # except Exception as e:
        #     print(e)

    def evaluate(self, gt_fname, rec_fname):
        # try:
        music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
        print(f"Music nDCG: {music_ndcg:.6}")
        print(f"Tag nDCG: {tag_ndcg:.6}")
        print(f"Score: {score:.6}")
        # except Exception as e:
        #     print(e)

def mid_check(q_dataloader, model, tmp_result_path, answer_file_path, id2song_dict, id2tag_dict, is_cuda, num_songs) :
    evaluator = ArenaEvaluator()
    device = 'cuda' if is_cuda else 'cpu'

    remove_file(tmp_result_path)

    elements =[]
    for idx, (_id, _data) in enumerate(q_dataloader) :
        with torch.no_grad() :
            _data = _data.to(device)
            output = model(_data)

        songs_input, tags_input = torch.split(_data, num_songs, dim=1)
        songs_output, tags_output = torch.split(output, num_songs, dim=1)

        songs_ids = binary_songs2ids(songs_input, songs_output, id2song_dict)
        tags_ids = binary_tags2ids(tags_input, tags_output, id2tag_dict)

        _id = list(map(int, _id))
        for i in range(len(_id)) :
            element = {'id':_id[i], 'songs':list(songs_ids[i]), 'tags':tags_ids[i]}
            elements.append(element)
    
    write_json(elements, tmp_result_path)
    evaluator.evaluate(answer_file_path, tmp_result_path)
    remove_file(tmp_result_path)


if __name__ == "__main__":
    fire.Fire(ArenaEvaluator)
