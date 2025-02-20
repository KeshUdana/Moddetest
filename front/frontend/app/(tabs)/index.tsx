import React, { useState } from 'react';
import { View, Button, Image, Text, StyleSheet, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import ParallaxScrollView from '@/components/ParallaxScrollView';

export default function HomeScreen() {
  const [image, setImage] = useState<string | null>(null);
  const [features, setFeatures] = useState<string[]>([]);

  // Image picker function
  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      const uri = result.assets[0].uri;
      setImage(uri);
      uploadImage(uri);
    } else {
      Alert.alert("No image selected", "Please select an image to continue.");
    }
  };

  // Upload image to Flask API
  const uploadImage = async (uri: string) => {
    const formData = new FormData();

    try {
      // Fetch the image and convert to Blob
      const response = await fetch(uri);
      const blob = await response.blob();

      // Append the image blob to FormData
      formData.append('file', blob, 'image.jpg');

      // Make the POST request to the API
      const res = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      // Handle the response from the API
      if (res.data.extracted_features) {
        setFeatures(res.data.extracted_features);
      } else {
        Alert.alert('Error', 'No features extracted from the image.');
      }
    } catch (error) {
      console.error('Upload failed', error);
      Alert.alert('Upload Failed', 'An error occurred while uploading the image.');
    }
  };

  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#A1CEDC', dark: '#1D3D47' }}
      headerImage={
        <Image
          source={require('@/assets/images/partial-react-logo.png')}
          style={styles.reactLogo}
        />
      }>
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">Welcome!</ThemedText>
      </ThemedView>
      <ThemedView style={styles.stepContainer}>
        <ThemedText type="subtitle">Step 1: Upload an image</ThemedText>
        <Button title="Pick an image" onPress={pickImage} />
      </ThemedView>
      {image && (
        <ThemedView style={styles.imageContainer}>
          <Image source={{ uri: image }} style={styles.image} />
          <Text style={styles.featuresText}>Features: {features.join(', ')}</Text>
        </ThemedView>
      )}
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  stepContainer: {
    gap: 8,
    marginBottom: 8,
  },
  imageContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  image: {
    width: 200,
    height: 200,
    marginTop: 10,
  },
  featuresText: {
    marginTop: 10,
    fontSize: 16,
    fontWeight: 'bold',
  },
  reactLogo: {
    height: 178,
    width: 290,
    bottom: 0,
    left: 0,
    position: 'absolute',
  },
});
