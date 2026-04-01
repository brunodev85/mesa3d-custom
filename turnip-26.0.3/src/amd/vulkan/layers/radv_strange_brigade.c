/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_cmd_buffer.h"
#include "radv_device.h"
#include "radv_entrypoints.h"

VKAPI_ATTR void VKAPI_CALL
strange_brigade_CmdPipelineBarrier2(VkCommandBuffer commandBuffer, const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);

   for (uint32_t i = 0; i < pDependencyInfo->imageMemoryBarrierCount; i++) {
      VkImageMemoryBarrier2 *barrier = (VkImageMemoryBarrier2 *)&pDependencyInfo->pImageMemoryBarriers[i];

      if (barrier->newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR &&
          barrier->srcAccessMask == VK_ACCESS_COLOR_ATTACHMENT_READ_BIT) {
         /* This game has a broken barrier right before present that causes rendering issues. Fix it
          * by modifying the src access mask.
          */
         barrier->srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
         break;
      }
   }

   device->layer_dispatch.app.CmdPipelineBarrier2(commandBuffer, pDependencyInfo);
}
