import { Controller, Post, UploadedFile, UseInterceptors } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import axios from 'axios';
import * as FormData from 'form-data';
import { Multer } from 'multer';


@Controller('upload')
export class UploadController {
  @Post()
  @UseInterceptors(FileInterceptor('file'))
  async uploadFile(@UploadedFile() file: Multer.File) {
    {
    if (!file) {
      return { error: 'No file uploaded' };
    }

    const formData = new FormData();
    formData.append('image', file.buffer, file.originalname);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: formData.getHeaders(),
      });

      return response.data;
    } catch (error) {
      return { error: 'Failed to process image', details: error.message };
    }
  }
}
}
