import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { HttpModule } from '@nestjs/axios';
import { AIService } from './ai.service';
import { ProductsController } from './products.controller';

@Module({
  imports: [HttpModule],
  controllers: [AppController,ProductsController],
  providers: [AppService,AIService],
  
  
})
export class AppModule {}
