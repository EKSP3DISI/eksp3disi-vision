import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload

class YouTubeUploader:
    def __init__(self, client_secrets_file):
        """
        Initialize YouTube uploader with OAuth credentials
        
        Args:
            client_secrets_file (str): Path to client secrets file from Google Cloud Console
        """
        self.client_secrets_file = client_secrets_file
        self.api_name = "youtube"
        self.api_version = "v3"
        self.scopes = ["https://www.googleapis.com/auth/youtube.upload"]
        self.credentials = None
        self.youtube = None

    def authenticate(self):
        """Authenticate with YouTube using OAuth 2.0"""
        try:
            # Get credentials and create API client
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                self.client_secrets_file, self.scopes)
            self.credentials = flow.run_local_server(port=0)
            
            self.youtube = googleapiclient.discovery.build(
                self.api_name, self.api_version, credentials=self.credentials)
            
            print("Authentication successful!")
            return True
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            return False

    def upload_video(self, video_file, title, description, privacy_status="private", 
                    category_id="22", tags=None):
        """
        Upload a video to YouTube
        
        Args:
            video_file (str): Path to video file
            title (str): Video title
            description (str): Video description
            privacy_status (str): Privacy status (private/public/unlisted)
            category_id (str): Video category ID (22 = People & Blogs)
            tags (list): List of video tags
        
        Returns:
            dict: Response from YouTube API or None if upload fails
        """
        if not os.path.exists(video_file):
            print(f"Video file not found: {video_file}")
            return None

        if not self.youtube:
            print("YouTube API client not initialized. Please authenticate first.")
            return None

        try:
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': tags or [],
                    'categoryId': category_id
                },
                'status': {
                    'privacyStatus': privacy_status,
                    'selfDeclaredMadeForKids': False,
                }
            }

            # Create MediaFileUpload object
            media = MediaFileUpload(
                video_file,
                mimetype='video/*',
                resumable=True
            )

            # Execute the upload request
            request = self.youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )

            print(f"Uploading {video_file}...")
            response = request.execute()
            
            print(f"Upload successful! Video ID: {response['id']}")
            print(f"Video URL: https://youtu.be/{response['id']}")
            
            return response

        except googleapiclient.errors.HttpError as e:
            print(f"An HTTP error occurred: {str(e)}")
            return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None