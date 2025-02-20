import { Controller, Post, UploadedFile, UseInterceptors } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { AIService } from './ai.service';

@Controller('products')
export class ProductsController {
  constructor(private readonly aiService: AIService) {}

  // Endpoint for retailers to upload product images
  @Post('upload')
  @UseInterceptors(FileInterceptor('file'))
  async uploadProduct(@UploadedFile() file: Express.Multer.File) {
    // Extract features from the uploaded image
    const features = await this.aiService.extractFeatures(file.buffer);

    // Save the features and metadata to MongoDB (not implemented in this barebones version)
    return { message: 'Product uploaded successfully', features };
  }

  // Endpoint for users to find similar products
  @Post('find-similar')
  @UseInterceptors(FileInterceptor('file'))
  async findSimilar(@UploadedFile() file: Express.Multer.File) {
    // Extract features from the user's image
    const features = await this.aiService.extractFeatures(file.buffer);

    // Compare features with products in MongoDB (not implemented in this barebones version)
    return { message: 'Similar products found', features };
  }
}