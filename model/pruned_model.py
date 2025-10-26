import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLPreTrainedModel
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .token_operations import TokenPruner, TokenFuser

class QwenVLPrunedModel(Qwen2VLPreTrainedModel):
    """带有令牌剪枝和融合的Qwen-VL模型"""
    
    def __init__(self, config, model_config):
        super().__init__(config)
        
        # 加载基础模型
        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_config.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # 初始化剪枝和融合模块
        self.token_pruner = TokenPruner(model_config.token_pruning)
        self.token_fuser = TokenFuser(model_config.token_fusion)
        
        # 统计信息
        self.pruning_stats = {
            'original_seq_lens': [],
            'pruned_seq_lens': [],
            'fused_seq_lens': []
        }
    
    def forward(self, input_ids: torch.Tensor, 
                pixel_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入令牌ID
            pixel_values: 像素值
            attention_mask: 注意力掩码
            labels: 标签
        
        Returns:
            模型输出
        """
        # 获取视觉特征
        vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        
        # 获取文本嵌入
        text_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)
        
        # 合并视觉和文本特征
        batch_size = input_ids.shape[0]
        image_seq_len = image_embeds.shape[1]
        text_seq_len = text_embeds.shape[1]
        
        # 创建图像注意力掩码
        image_attention_mask = torch.ones(
            (batch_size, image_seq_len), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        
        # 合并嵌入和注意力掩码
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        
        # 记录原始序列长度
        original_seq_len = combined_embeds.shape[1]
        self.pruning_stats['original_seq_lens'].append(original_seq_len)
        
        # 应用令牌剪枝
        pruned_embeds, pruned_attention_mask = self.token_pruner(
            combined_embeds, combined_attention_mask
        )
        pruned_seq_len = pruned_embeds.shape[1]
        self.pruning_stats['pruned_seq_lens'].append(pruned_seq_len)
        
        # 应用令牌融合
        fused_embeds, fused_attention_mask = self.token_fuser(
            pruned_embeds, pruned_attention_mask
        )
        fused_seq_len = fused_embeds.shape[1]
        self.pruning_stats['fused_seq_lens'].append(fused_seq_len)
        
        # 通过语言模型
        outputs = self.base_model.language_model(
            inputs_embeds=fused_embeds,
            attention_mask=fused_attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def generate(self, input_ids: torch.Tensor,
                pixel_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """生成方法"""
        # 类似forward的处理流程，但用于生成任务
        vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        
        text_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)
        
        batch_size = input_ids.shape[0]
        image_seq_len = image_embeds.shape[1]
        
        image_attention_mask = torch.ones(
            (batch_size, image_seq_len), 
            dtype=attention_mask.dtype, 
            device=attention_mask.device
        )
        
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        
        # 应用剪枝和融合
        pruned_embeds, pruned_attention_mask = self.token_pruner(
            combined_embeds, combined_attention_mask
        )
        fused_embeds, fused_attention_mask = self.token_fuser(
            pruned_embeds, pruned_attention_mask
        )
        
        # 生成
        return self.base_model.language_model.generate(
            inputs_embeds=fused_embeds,
            attention_mask=fused_attention_mask,
            **kwargs
        )
    
    def get_pruning_statistics(self) -> Dict[str, float]:
        """获取剪枝统计信息"""
        if not self.pruning_stats['original_seq_lens']:
            return {}
        
        original_lens = torch.tensor(self.pruning_stats['original_seq_lens'])
        pruned_lens = torch.tensor(self.pruning_stats['pruned_seq_lens'])
        fused_lens = torch.tensor(self.pruning_stats['fused_seq_lens'])
        
        stats = {
            'avg_original_seq_len': original_lens.float().mean().item(),
            'avg_pruned_seq_len': pruned_lens.float().mean().item(),
            'avg_fused_seq_len': fused_lens.float().mean().item(),
            'pruning_ratio': (1 - pruned_lens.float().mean() / original_lens.float().mean()).item(),
            'compression_ratio': (original_lens.float().mean() / fused_lens.float().mean()).item()
        }
        
        # 重置统计
        self.pruning_stats = {k: [] for k in self.pruning_stats.keys()}
        
        return stats