#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:
    # セクション1: コア設定と初期化
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """スケールと回転から 3D ガウシアンの共分散を作る。

            計算式:
              1) `s' = scaling_modifier * scaling`
              2) `L = R(q) @ diag(s'_x, s'_y, s'_z)`
              3) `Sigma = L @ L^T`
              4) 対称行列 `Sigma` を 6 要素 `[xx, xy, xz, yy, yz, zz]` に圧縮して返す

            ここで `R(q)` はクォータニオン `rotation` から作る回転行列。
            `Sigma = L L^T` にすることで、共分散は常に対称・半正定値になる。

            具体例（1点）:
              - `scaling = [2, 1, 1]`, `scaling_modifier = 1`
              - z 軸まわり 90 度回転:
                  `R = [[0,-1,0],[1,0,0],[0,0,1]]`
              - `diag(s) = [[2,0,0],[0,1,0],[0,0,1]]`
              - `L = R @ diag(s) = [[0,-1,0],[2,0,0],[0,0,1]]`
              - `Sigma = L L^T = [[1,0,0],[0,4,0],[0,0,1]]`

            解釈:
              - 回転前は x 方向に長軸（分散 4）を持つ楕円体。
              - 90 度回すと長軸は y 方向へ移り、共分散の大きい対角成分も
                `xx` から `yy` へ移る。
            """
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    # セクション2: シーン生成と学習のエントリポイント
    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # セクション3: 適応的トポロジ更新（densify / prune）
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        """勾配統計に基づいてガウシアンを増減し、表現能力と計算効率を調整する。

        判断材料:
          - 位置勾配統計: `grads = xyz_gradient_accum / denom`
          - 点のワールドスケール: `get_scaling.max(dim=1).values`
          - 点のスクリーン半径: `max_radii2D`（外部から渡される `radii` を一時保持）
          - 点の不透明度: `get_opacity`
          - シーン基準長: `extent`、密化境界係数: `percent_dense`

        densify 条件:
          - clone (`densify_and_clone`):
            `||grads|| >= max_grad` かつ
            `max_scaling <= percent_dense * extent`
            （比較的小さいガウシアンを複製）
          - split (`densify_and_split`):
            `grad >= max_grad` かつ
            `max_scaling > percent_dense * extent`
            （比較的大きいガウシアンを分割）

        prune 条件:
          - 基本: `opacity < min_opacity`
          - 追加（`max_screen_size` が有効なとき）:
            `max_radii2D > max_screen_size` または
            `max_scaling > 0.1 * extent`

        具体的な操作:
          1) `tmp_radii = radii` を保持し、clone と split を順に実行する。
          2) clone は選択点の属性（位置・特徴・不透明度・スケール・回転）を
             そのまま複製して末尾に連結する。
          3) split は選択点を N 個に分割し、スケールを分配して回転付きの正規サンプルで
             新しい位置を生成、属性を複製して追加後、元点を削除する。
          4) 追加・削除時は optimizer の Parameter と内部状態（exp_avg, exp_avg_sq）も
             同期的に更新し、点数変更後に統計バッファを再初期化する。
          5) 最後に prune 条件で不要点を削除し、`tmp_radii` を解放して CUDA キャッシュを空ける。
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # セクション4: シリアライズとチェックポイント入出力
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        """`capture()` で保存した学習状態を読み戻し、学習を再開可能にする。

        train.py では以下の流れで本メソッドに引数が渡される:
          1) 保存時: `torch.save((gaussians.capture(), iteration), chkpnt_path)`
          2) 復元時: `(model_params, first_iter) = torch.load(checkpoint)`
          3) 呼び出し: `gaussians.restore(model_params, opt)`
        ここで `model_params` が本メソッドの `model_args`、`opt` が `training_args` に対応する。

        なぜこの復元で十分か:
          - ガウシアン内部表現:
            位置・SH特徴・スケール・回転・不透明度を
            直接 Parameter として復元するため、レンダリング状態は再現できる。
          - 構造適応の内部状態:
            `max_radii2D`, `xyz_gradient_accum`, `denom` を復元するため、
            densification/pruning の判断履歴を引き継げる。
          - 学習内部設定:
            `training_setup(training_args)` で optimizer の param group と scheduler を再構築し、
            その後 `optimizer.load_state_dict(opt_dict)` で Adam のモーメント等を復元するため、
            最適化の連続性を保てる。`spatial_lr_scale` も復元済み値を使う。

        """
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # セクション5: 実行時状態アクセスと補助関数
    def reset_opacity(self):
        """不透明度を低めにリセットし、densification/pruning の再評価を促す。

        呼び出しタイミング（train.py）:
          - `iteration % opt.opacity_reset_interval == 0`
          - または `dataset.white_background` かつ
            `iteration == opt.densify_from_iter`
          - ただし上記は `iteration < opt.densify_until_iter` の範囲内でのみ実行される。

        なぜ必要か:
          - 学習中に opacity が飽和すると、少数の点が寄与を抱え込み、
            新しい点の成長（densify）や不要点の淘汰（prune）が進みにくくなる。
          - 一度 opacity を低めに戻すことで、可視化寄与の再配分を起こし、
            点群の再編（増殖/削除）を安定して進めやすくする。

        実装上の操作:
          - 現在の opacity を `0.01` 上限でクリップし、内部表現に逆写像した値で
            `opacity` Parameter を置換する。
          - 置換時に optimizer 側の状態テンソル形状も整合させる。
        """
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        """画像名に対応する露出補正行列（3x4アフィン変換）を返す。

        exposure とは:
          - カメラ/画像ごとの明るさ・色ずれを吸収するための補正パラメータ。
          - 各画像に 1 つの `3x4` 行列を割り当て、RGB に対する線形変換＋バイアス
            （アフィン変換）として扱う。
          - `self._exposure` の実体は形状 `(num_images, 3, 4)` の Parameter。

        `self.exposure_mapping[image_name]` の実体:
          - `create_from_pcd` で作られる `dict[str, int]`。
          - キーは `cam_info.image_name`、値はその画像の exposure 行列インデックス。
          - したがって通常は `self._exposure[self.exposure_mapping[image_name]]` で
            画像ごとの行列を取り出す。

        事前学習済み露出が読み込まれている場合は、`self.pretrained_exposures` から
        固定値（非学習）を返す。
        """
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # セクション6: PLY属性スキーマ補助
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # セクション7: densificationの内部プリミティブ
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """小さいガウシアンを勾配に基づいて複製（clone）する。

        ワークフロー:
          1) 候補抽出:
             `torch.norm(grads, dim=-1) >= grad_threshold`
             で位置勾配ノルムが大きい点を選ぶ。
          2) 形状条件で絞り込み:
             `max_scaling <= percent_dense * scene_extent`
             を満たす「比較的小さい」点だけ残す。
          3) `selected_pts_mask` で各内部表現（xyz/features/opacity/scale/rotation）
             を抽出し、そのまま新規点として追加する。
          4) `densification_postfix` を通して optimizer 状態と内部バッファを更新する。

        条件の意図:
          - 勾配ノルムが大きい点は、現在の表現で誤差に強く寄与しており、
            近傍に点を増やす価値が高い。
          - そのうち「小さい点」のみ clone するのは、既に細かい表現を持つ領域で
            密度を上げて解像度を高めるため。大きい点は clone ではなく split 側に回し、
            役割分担して過密化や不安定化を避ける。
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """大きいガウシアンを N 個へ分割して再配置する。

        ワークフロー:
          1) 勾配条件で候補抽出:
             `padded_grad >= grad_threshold` を満たす点を選ぶ。
             （`grads` が初期点数より短い可能性に備えて `padded_grad` を使う）
          2) 形状条件で絞り込み:
             `max_scaling > percent_dense * scene_extent` の「大きい点」のみ残す。
          3) 子中心のサンプリング:
             - 対象点のスケールを標準偏差 `stds` として `N` 回複製
             - ローカル座標で `Normal(0, stds)` からオフセット `samples` を生成
             - 親の回転を適用し、親中心へ加算して `new_xyz` を得る
          4) 子属性の構築:
             - 特徴・不透明度・回転は親を複製
             - スケールは `get_scaling / (0.8 * N)` 相当に縮小し、子を小さくする
          5) 追加と置換:
             - `densification_postfix` で子を追加
             - 直後に `prune_filter` で親点を削除
             - 結果として「1個の大きい親」を「N個の小さい子」に置き換える

        この分割により、粗い表現を局所的に高解像度化しつつ、親の向きと分布傾向を
        保ったまま点群を再編できる。
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_tmp_radii,
    ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # セクション8: Optimizerパラメータ/状態の整合ユーティリティ
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def replace_tensor_to_optimizer(self, tensor, name):
        """指定グループの Parameter を新テンソルで置換し、optimizer状態を整合させる。

        なぜ必要か:
          - `reset_opacity` のように値を大きく作り直す処理では、既存 Parameter を
            そのまま上書きするより、新しい Parameter に差し替える方が安全な場合がある。
          - ただし Adam は Parameter オブジェクトをキーに state を保持するため、
            差し替え後に古い state を参照すると更新が不整合になる。

        この関数が行うこと:
          - 対象グループ（例: `opacity`）の旧 state を取得
          - `exp_avg` / `exp_avg_sq` を `zeros_like(tensor)` で再初期化
          - param group の Parameter を新しいものへ差し替え
          - state のキーを新 Parameter に付け替え

        これにより、Parameter 本体と optimizer 内部状態の対応関係を保ったまま、
        該当パラメータだけモーメント履歴をリセットして学習を継続できる。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        """各 param group に新規テンソルを連結し、optimizer 状態も拡張する。

        `replace_tensor_to_optimizer` との違い:
          - replace: 既存 Parameter 全体を新テンソルに「置換」する。
            （例: opacity リセット時）
          - cat（本関数）: 既存 Parameter の末尾に新規行を「追加連結」する。
            （例: densify で点数が増えるとき）

        cat が必要な理由:
          - densify では既存点は保持しつつ、新しい点だけ増やしたい。
          - そのため Parameter を `torch.cat` で拡張し、同時に Adam の
            `exp_avg` / `exp_avg_sq` も追加分をゼロ初期化して同じ先頭次元で拡張する。
          - これにより、旧点のモーメント履歴は維持しつつ、新点は履歴なしで学習を開始できる。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
