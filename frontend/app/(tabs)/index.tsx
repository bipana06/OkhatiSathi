import React, { useState, useEffect } from 'react';
import { Button, View, StyleSheet, Text, ActivityIndicator, Alert, Image, Platform } from 'react-native';
import { Audio } from 'expo-av';
import * as ImagePicker from 'expo-image-picker';
import { getMedicineAudio, processMedicineImage } from '../apiServices'; 


interface PhotoState {
  uri: string;
  type: string; 
  name: string; 
  base64?: string; 
}

export default function HomeScreen() {
  const [identifiedMedicineName, setIdentifiedMedicineName] = useState<string | null>(null);
  const [audioUri, setAudioUri] = useState<string | null>(null);
  const [sound, setSound] = useState<Audio.Sound | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [photo, setPhoto] = useState<PhotoState | null>(null); // Use the interface
  const [isPlaying, setIsPlaying] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null); // New state for status messages

  const handleImageUpload = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    // if (!permissionResult.granted) {
    //   Alert.alert('Permission Required', 'Permission to access camera roll is required to upload images.');
    //   return;
    // }

    try {
      const response = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false, 
        quality: 0.7, 
        base64: true, 
      });

      if (!response.canceled && response.assets && response.assets.length > 0) {
        const selectedAsset = response.assets[0];
        if (selectedAsset.uri) {
          const uriParts = selectedAsset.uri.split('/');
          const fileName = uriParts[uriParts.length - 1];
          let fileType = selectedAsset.type || 'image/jpeg'; // Default type
          if (fileName) {
              const fileExt = fileName.split('.').pop();
              if (fileExt) {
                  fileType = `image/${fileExt === 'jpg' ? 'jpeg' : fileExt}`;
              }
          }

          setPhoto({
            uri: selectedAsset.uri,
            type: fileType,
            name: fileName || 'uploaded_image.jpg', 
            base64: selectedAsset.base64 ?? undefined,
          });
          // Reset previous results when a new image is selected
          setIdentifiedMedicineName(null);
          setAudioUri(null);
          setError('');
          handleStopAudio(); 
        }
      }
    } catch (err) {
        console.error("Error picking image:", err);
        Alert.alert("Error", "Could not select image.");
        setError("Failed to select image.");
    }
  };

  const triggerProcessImage = async () => {
    if (!photo) {
      Alert.alert("No Image", "Please upload an image first.");
      return;
    }

    if (!photo.base64) {
         if (Platform.OS !== 'web') {
            console.warn("Base64 not found in picker result on native, trying FileSystem...");
            setError("Could not get image data (base64 missing). Try re-selecting the image.");
            return; 
         } else {
             console.error("Base64 string is missing from the selected photo object on web.");
             Alert.alert("Error", "Could not get image data. Please try selecting the image again.");
             setError("Could not get image data.");
             return; // Stop processing
         }

    }
    const base64Image = photo.base64;

    setLoading(true);
    setError('');
    setIdentifiedMedicineName(null);
    setAudioUri(null);
    setStatusMessage("Processing image... Please wait."); // Set status message
    handleStopAudio();

    try {
      console.log('Sending base64 image to backend...');
      const response = await processMedicineImage(base64Image); // Pass the base64 string

      if (response.drug_name) {
        console.log('Medicine identified:', response.drug_name);
        setStatusMessage("Medicine identified. Preparing audio..."); // Update status message
        setIdentifiedMedicineName(response.matching_name);
        handleFetchAudio(response.drug_name);
      } else {
        setError(response.message || 'Medicine name not identified in the image.');
        setStatusMessage(null); // Clear status message
      }
    } catch (err: any) {
      console.error('Error while processing image:', err);
      setError(`Failed to process image: ${err.message || 'Please try again.'}`);
      setStatusMessage(null); // Clear status message
    } finally {
      setLoading(false);
    }
  };

  const handleFetchAudio = async (medicineName: string) => {
    setError('');
    setStatusMessage("Fetching audio... Please wait."); // Update status message
    try {
      const audioFileUrl = await getMedicineAudio(medicineName.toUpperCase());
      if (audioFileUrl) {
        await Audio.setAudioModeAsync({ playsInSilentModeIOS: true });
  
        const { sound: newSound } = await Audio.Sound.createAsync(
          { uri: audioFileUrl },
          { shouldPlay: true }
        );
  
        setSound(newSound);
        setAudioUri(audioFileUrl); 
        setIsPlaying(true); // Start as playing
        setStatusMessage("Audio Playing..."); // Update status message
  
        newSound.setOnPlaybackStatusUpdate((status) => {
          if (status.isLoaded && status.didJustFinish) {
            handleStopAudio(); // Stop and unload when done
          }
        });
  
      } else {
        setError('Failed to retrieve audio URL.');
      }
    } catch (err: any) {
      setError(`Failed to fetch or play audio: ${err.message || 'Please try again.'}`);
      setAudioUri(null); 
      setStatusMessage(null); // Clear status message
    }
  };
  
  const togglePlayback = async () => {
    if (sound) {
      const status = await sound.getStatusAsync();
      if (status.isLoaded) {
        if (status.isPlaying) {
          await sound.pauseAsync();
          setIsPlaying(false);
          setStatusMessage("Audio Paused, Hit Play to continue"); // Update status message
        } else {
          await sound.playAsync();
          setIsPlaying(true);
          setStatusMessage("Audio Playing..."); // Update status message
        }
      }
    }
  };

  const handleStopAudio = async () => {
    if (sound) {
      try {
        await sound.pauseAsync();
        await sound.unloadAsync();
      } catch (error) {
        console.error("Error stopping/unloading sound:", error);
      } finally {
        setSound(null);
        setAudioUri(null);
        setIsPlaying(false);
        setStatusMessage(null); // Clear status message
        setIdentifiedMedicineName(null);
      }
    }
  };


  useEffect(() => {
    return () => {
      handleStopAudio();
    };
  }, [sound]); 

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome to OkhatiSathi!</Text>

      {/* Image Display */}
      {photo && (
        <View style={styles.imageContainer}>
            <Image source={{ uri: photo.uri }} style={styles.image} resizeMode="contain" />
        </View>
      )}

       {/* Buttons */}
      <View style={styles.buttonContainer}>
        {photo && !loading && (
            <Button title="Process Image" onPress={triggerProcessImage} disabled={loading} />
        )}
        <Button 
          title={photo ? "Upload Another Image" : "Upload Medicine Image"} 
          onPress={handleImageUpload} 
        />
      </View>


      {/* Loading Indicator */}
      {loading && <ActivityIndicator size="large" color="#4CAF50" style={styles.loadingIndicator} />}

      {/* Error Message */}
      {error && <Text style={styles.errorText}>{error}</Text>}

      {/* Identified Medicine Name */}
      {identifiedMedicineName && !loading && (
        <Text style={styles.medicineName}>
          <Text style={styles.labelText}>Identified Medicine: </Text>
          <Text style={styles.medicineNameValue}>{identifiedMedicineName}</Text>
        </Text>      
      )}

      {/* Status Message */}
      {statusMessage && (
        <Text style={styles.statusMessage}>{statusMessage}</Text>
      )}

      {/* Audio Playback */}
      {audioUri && !loading && (
      <View style={styles.audioControls}>
        <View style={styles.audioButtonRow}>
          <Button title={isPlaying ? "Pause" : "Play"} onPress={togglePlayback} />
          <View style={{ width: 10 }} /> {/* Spacer */}
          <Button title="Stop" onPress={handleStopAudio} color="#FF5733" />
        </View>
      </View>
    )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#333',
  },
  imageContainer: {
    marginVertical: 20,
    width: '90%',
    height: 250,
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 10,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  buttonContainer: {
    gap: 10,
    marginTop: 10,
    width: '90%',
    color: '#3D365C',
  },
  loadingIndicator: {
    marginTop: 20,
  },
  errorText: {
    color: 'red',
    marginTop: 10,
    textAlign: 'center',
  },
  medicineName: {
    fontSize: 18,
    fontWeight: '600',
    marginTop: 20,
    color: '#4CAF50',
  },
  audioControls: {
    marginTop: 20,
    alignItems: 'center',
    gap: 10,
  },
  audioButtonRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  statusMessage: {
    marginTop: 10,
    fontSize: 16,
    color: '#555',
    textAlign: 'center',
  },
  labelText: {
    color: '#424769', // Color for "Identified Medicine:"
  },
  medicineNameValue: {
    color: '#99627A', // Color for the identified medicine name
    fontWeight: '800',
    fontSize: 22,
    fontStyle: 'italic',

  },
});
