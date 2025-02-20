import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';

@Injectable()
export class AIService {
  constructor(private readonly httpService: HttpService) {}

  async extractFeatures(image: Buffer): Promise<number[]> {
    // Create a FormData object
    const formData = new FormData();
    formData.append('file', new Blob([image]), 'image.jpg');

    // Call the FastAPI feature extraction service
    const response = await firstValueFrom(
      this.httpService.post(
        'https://feature-extraction-service-xyz.a.run.app/extract_features',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        },
      ),
    );

    // Return the extracted features
    return response.data.features;
  }
}