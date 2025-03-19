/**
  ******************************************************************************
  * @file    predictive_maintenance_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-03-18T00:33:43+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef PREDICTIVE_MAINTENANCE_DATA_PARAMS_H
#define PREDICTIVE_MAINTENANCE_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_PREDICTIVE_MAINTENANCE_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_predictive_maintenance_data_weights_params[1]))
*/

#define AI_PREDICTIVE_MAINTENANCE_DATA_CONFIG               (NULL)


#define AI_PREDICTIVE_MAINTENANCE_DATA_ACTIVATIONS_SIZES \
  { 384, }
#define AI_PREDICTIVE_MAINTENANCE_DATA_ACTIVATIONS_SIZE     (384)
#define AI_PREDICTIVE_MAINTENANCE_DATA_ACTIVATIONS_COUNT    (1)
#define AI_PREDICTIVE_MAINTENANCE_DATA_ACTIVATION_1_SIZE    (384)



#define AI_PREDICTIVE_MAINTENANCE_DATA_WEIGHTS_SIZES \
  { 10516, }
#define AI_PREDICTIVE_MAINTENANCE_DATA_WEIGHTS_SIZE         (10516)
#define AI_PREDICTIVE_MAINTENANCE_DATA_WEIGHTS_COUNT        (1)
#define AI_PREDICTIVE_MAINTENANCE_DATA_WEIGHT_1_SIZE        (10516)



#define AI_PREDICTIVE_MAINTENANCE_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_predictive_maintenance_activations_table[1])

extern ai_handle g_predictive_maintenance_activations_table[1 + 2];



#define AI_PREDICTIVE_MAINTENANCE_DATA_WEIGHTS_TABLE_GET() \
  (&g_predictive_maintenance_weights_table[1])

extern ai_handle g_predictive_maintenance_weights_table[1 + 2];


#endif    /* PREDICTIVE_MAINTENANCE_DATA_PARAMS_H */
