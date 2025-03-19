
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "predictive_maintenance.h"
#include "predictive_maintenance_data.h"

/* USER CODE BEGIN includes */
 extern UART_HandleTypeDef huart2;

 #define BYTES_IN_FLOATS 5*4

 #define TIMEOUT 1000

 #define SYNCHRONISATION 0xAB

 #define ACKNOWLEDGE 0xCD

 #define CLASS_NUMBER 5

  void synchronize_UART(void);
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_PREDICTIVE_MAINTENANCE_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_PREDICTIVE_MAINTENANCE_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_PREDICTIVE_MAINTENANCE_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_PREDICTIVE_MAINTENANCE_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_PREDICTIVE_MAINTENANCE_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_PREDICTIVE_MAINTENANCE_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_PREDICTIVE_MAINTENANCE_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_PREDICTIVE_MAINTENANCE_OUT_NUM] = {
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_PREDICTIVE_MAINTENANCE_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle predictive_maintenance = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_predictive_maintenance_create_and_init(&predictive_maintenance, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_predictive_maintenance_create_and_init");
    return -1;
  }

  ai_input = ai_predictive_maintenance_inputs_get(predictive_maintenance, NULL);
  ai_output = ai_predictive_maintenance_outputs_get(predictive_maintenance, NULL);

#if defined(AI_PREDICTIVE_MAINTENANCE_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_PREDICTIVE_MAINTENANCE_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_PREDICTIVE_MAINTENANCE_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_PREDICTIVE_MAINTENANCE_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_PREDICTIVE_MAINTENANCE_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_PREDICTIVE_MAINTENANCE_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_predictive_maintenance_run(predictive_maintenance, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_predictive_maintenance_get_error(predictive_maintenance),
        "ai_predictive_maintenance_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */

int acquire_and_process_data(ai_i8 *data[])//peut être essayer avec ai_i32
{
    //
    // 1. Variables for data acquisition
    //
    unsigned char tmp[BYTES_IN_FLOATS] = {0};
    int num_elements = sizeof(tmp) / sizeof(tmp[0]);//A verifier qu'on ait bien 20
    int num_floats = num_elements / 4;
    //
    // 2. Receive data from UART
    //
    HAL_StatusTypeDef status = HAL_UART_Receive(&huart2, (uint8_t *)tmp, sizeof(tmp), HAL_MAX_DELAY);
    // Check the return status of HAL_UART_Receive
    if (status != HAL_OK)
    {
      printf("Failed to receive data from UART. Error code: %d\n", status);
      return (1);
    }
    //
    // 3. Reconstruct floats from bytes
    //
    float value;
    if (num_elements % 4 != 0)
    {
      printf("The array length is not a multiple of 4 bytes. Cannot reconstruct floats.\n");
      return (1);
    }
    for (size_t i = 0; i < num_floats; i++)
    {
      unsigned char bytes[4] = {0};
      // Reconstruction of the bytes
      for (size_t j = 0; j < 4; j++)
      {
        bytes[j] = tmp[i * 4 + j];
      }



      // Store the bytes in 'data'
      for (size_t k = 0; k < 4; k++)
		{
		  ((uint8_t *)data)[(i * 4 + k)] = bytes[k];

		}

	   //value = *((float*)(data + i * 4));  // Cast des 4 octets en float
	  }


    return (0);
}

int post_process(ai_i8 *data[])
{
    //
    // Get the output data
    //
    if (data == NULL)
    {
      printf("The output data is NULL.\n");
      return (1);
    }

    uint8_t unique_label = 5;
    uint8_t *output = data;
    // An array to store the float outputs
    float outs[CLASS_NUMBER] = {0.0};
    uint8_t outs_uint8[CLASS_NUMBER] = {0};
    /* Convert the probability to float */
    for (size_t i = 0; i < CLASS_NUMBER; i++)
    {
      uint8_t temp[4] = {0};
      // Extract 4 bytes to reconstruct a float
      for (size_t j = 0; j < 4; j++)
      {
        temp[j] = output[i * 4 + j];
      }
      // Reconstruct the float from the bytes
      outs[i] = *(float *)&temp;

      //
	  // Transform the multi class data in one output
	  //
      if(outs[i] > outs[unique_label])
          		unique_label = i;

      // Convert the float to uint8_t for UART transmission
      //outs_uint8[i] = (char)(outs[i] * 255);
    }

    //
	// Transmit the output data
	//
    float to_transmit = *((float*)(data + 0 * 4));//unique_label;
    //uint8_t byte_array[4];

    // Copier les 4 octets du float dans le tableau byte_array
    //memcpy(byte_array, &to_transmit, sizeof(to_transmit));
	HAL_StatusTypeDef status = HAL_UART_Transmit(&huart2, &unique_label, sizeof(unique_label), TIMEOUT);
	// Check the return status of HAL_UART_Transmit
	if (status != HAL_OK)
	{
		printf("Failed to transmit data to UART. Error code: %d\n", status);
		return (1);
	}
	return 0;
}

void synchronize_UART(void)

{

    bool is_synced = 0;

    unsigned char rx[2] = {0};

    unsigned char tx[2] = {ACKNOWLEDGE, 0};

    while (!is_synced)

    {

      HAL_UART_Receive(&huart2, (uint8_t *)rx, sizeof(rx), TIMEOUT);

      if (rx[0] == SYNCHRONISATION)

      {

        HAL_UART_Transmit(&huart2, (uint8_t *)tx, sizeof(tx), TIMEOUT);

        is_synced = 1;

      }

    }

    return;

}

/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  printf("\r\nTEMPLATE - initialization\r\n");

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
  int res = -1;
  uint8_t *in_data = ai_input[0].data;
  uint8_t *out_data = ai_output[0].data;
  synchronize_UART();

  if (predictive_maintenance) {
    do {
      /* 1 - acquire and pre-process input data */
      res = acquire_and_process_data(in_data);
      //float default_values[5] = {298.45421655, 308.70043998, 2679.36885837, 10.65600098, 84.97335622};// Y = 3 from index 5
      //float default_values[5] = {302.48711493, 311.36830076, 1626.36601512, 33.3840344, 229.31699244};// Y = 1 from index 8
      float default_values[5] = {300.43377048,  310.41191899, 1352.47020251,   56.86952747,  209.45033752};// Y = 4 from index 7
      memcpy(in_data, default_values, BYTES_IN_FLOATS);
      /* 2 - process the data - call inference engine */
      if (res == 0)
        res = ai_run();
      /* 3- post-process the predictions */
      if (res == 0)
      {
        res = post_process(out_data);
    	  //HAL_StatusTypeDef status = HAL_UART_Transmit(&huart2, (uint8_t *)in_data, 4, TIMEOUT);
    	  //float my_value = 298.45421655;
    	  //HAL_UART_Transmit(&huart2, (uint8_t*)&my_value, sizeof(my_value), HAL_MAX_DELAY);
      }
    } while (res==0);
  }
  if (res) {
    ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
    ai_log_err(err, "Process has FAILED");
  }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
