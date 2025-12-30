from pathlib import Path
import pandas as pd
import numpy as np

def cnn_postprocess(result: list, out_dir: Path) -> None:
    """
    Postprocess CNN result
    """
    results = []
    for res in result:
        mask_prob, kps_pred, im_names, orig_sizes, infer_sizes = res
        if kps_pred is None:
            raise ValueError('cnn_postprocess expects keypoints predictions, but got None')
        preds = kps_pred.cpu().numpy()
        df1 = pd.DataFrame(
            np.hstack([
                np.array(im_names).reshape(-1, 1),
                orig_sizes,
                infer_sizes
            ]),
            columns=['filepath', 'height_orig', 'width_orig', 'height_infer', 'width_infer']
        )
        df2 = pd.DataFrame(preds,
            columns=np.concatenate([[f'x{i+1}', f'y{i+1}'] for i in range(4)]).tolist()
        )
        df = pd.concat([df1, df2], axis=1)
        results.append(df)

    results = pd.concat(results)
    results.to_csv(out_dir / 'results.csv', index=False)



# В будущем поменять на постобработку масок
def krcnn_postprocess(result: list, out_dir: Path) -> None:
    """
    Postprocess k-R-CNN result
    """
    results = []
    for res in result:
        preds, im_names, orig_sizes, infer_sizes = res
        preds = [p.cpu().numpy() for p in preds]

        # Align shape of preds coords
        predicts = []
        for p in preds:
            if p.shape[0] < 4:
                p = np.vstack([p, np.zeros((4 - p.shape[0], p.shape[1]), dtype=p.dtype)])
            predicts.append(p.reshape(-1))

        df1 = pd.DataFrame(
            np.hstack([
                np.array(im_names).reshape(-1, 1),
                orig_sizes,
                infer_sizes
            ]),
            columns=['filepath', 'height_orig', 'width_orig', 'height_infer', 'width_infer']
        )
        df2 = pd.DataFrame(predicts,
            columns=np.concatenate([[f'x{i+1}', f'y{i+1}'] for i in range(4)]).tolist()
        )
        df = pd.concat([df1, df2], axis=1)
        results.append(df)

    results = pd.concat(results)
    results.to_csv(out_dir / 'results.csv', index=False)