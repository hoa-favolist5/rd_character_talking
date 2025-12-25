# AWS Services Setup Guide

Complete guide to set up all AI services used in the Character AI project.

## Services Overview

| Service | Purpose | Required |
|---------|---------|----------|
| **Amazon Bedrock** | LLM (Claude 3) for conversation | ✅ Yes |
| **Amazon Transcribe** | Speech-to-Text (Japanese) | ✅ Yes |
| **Amazon Polly** | Text-to-Speech (Japanese) | ✅ Yes |
| **Amazon S3** | Audio file storage | ✅ Yes |
| **Amazon RDS** | PostgreSQL database (production) | For EC2 only |

---

# Part 1: Local Development Setup

## Prerequisites

- AWS Account with billing enabled
- AWS CLI v2 installed
- Python 3.11+
- Docker Desktop

---

## Step 1: Install AWS CLI

### macOS
```bash
# Using Homebrew
brew install awscli

# Verify installation
aws --version
```

### Linux
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify
aws --version
```

### Windows
Download and run: https://awscli.amazonaws.com/AWSCLIV2.msi

---

## Step 2: Create IAM User for Development

### 2.1 Go to IAM Console

1. Sign in to AWS Console
2. Search for **IAM** → Click **IAM**
3. Click **Users** → **Create user**

### 2.2 Create User

```
┌─────────────────────────────────────────────────────────────┐
│  Create user                                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User name: character-ai-dev                                │
│                                                             │
│  ☐ Provide user access to the AWS Management Console       │
│    (Not needed for programmatic access)                     │
│                                                             │
│                                            [Next]           │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Attach Policies

Select **Attach policies directly** and add these policies:

| Policy Name | Purpose |
|-------------|---------|
| `AmazonBedrockFullAccess` | Access to Bedrock models |
| `AmazonS3FullAccess` | Store/retrieve audio files |
| `AmazonPollyFullAccess` | Text-to-Speech |
| `AmazonTranscribeFullAccess` | Speech-to-Text |

```
┌─────────────────────────────────────────────────────────────┐
│  Set permissions                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ● Attach policies directly                                 │
│                                                             │
│  Search: [                                    ]             │
│                                                             │
│  ☑️ AmazonBedrockFullAccess                                 │
│  ☑️ AmazonS3FullAccess                                      │
│  ☑️ AmazonPollyFullAccess                                   │
│  ☑️ AmazonTranscribeFullAccess                              │
│                                                             │
│                                            [Next]           │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Create Access Keys

1. Click the created user → **Security credentials**
2. Scroll to **Access keys** → **Create access key**
3. Select **Application running outside AWS**
4. Click **Create access key**
5. **Download .csv** or copy both keys immediately!

```
┌─────────────────────────────────────────────────────────────┐
│  ⚠️ Save these credentials - shown only once!               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Access key ID:     AKIAIOSFODNN7EXAMPLE                   │
│  Secret access key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY│
│                                                             │
│                                  [Download .csv file]       │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 3: Configure AWS CLI

```bash
aws configure
```

Enter your credentials:
```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: ap-northeast-1
Default output format [None]: json
```

Verify configuration:
```bash
aws sts get-caller-identity
```

Expected output:
```json
{
    "UserId": "AIDAXXXXXXXXXXXXXXXXX",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/character-ai-dev"
}
```

---

## Step 4: Set Up Amazon Bedrock

### 4.1 Enable Claude Model Access

1. Go to **Amazon Bedrock** console
2. Select region: **ap-northeast-1** (Tokyo)
3. Click **Model catalog** in sidebar
4. Find **Claude 3 Sonnet** → Click on it
5. Click **Request model access** or **Complete usage form**
6. Fill in the Anthropic usage form
7. Submit and wait for instant approval

### 4.2 Test Bedrock Access

```bash
# Test listing models
aws bedrock list-foundation-models \
    --region ap-northeast-1 \
    --query "modelSummaries[?contains(modelId, 'claude')].modelId"

# Test invoking Claude
aws bedrock-runtime invoke-model \
    --model-id anthropic.claude-3-sonnet-20240229-v1:0 \
    --region ap-northeast-1 \
    --content-type application/json \
    --body '{"anthropic_version":"bedrock-2023-05-31","max_tokens":50,"messages":[{"role":"user","content":"Say hello in Japanese"}]}' \
    /dev/stdout | jq -r '.content[0].text'
```

---

## Step 5: Create S3 Bucket for Audio

### 5.1 Create Bucket

```bash
# Create bucket (name must be globally unique)
aws s3 mb s3://character-audio-$(aws sts get-caller-identity --query Account --output text) \
    --region ap-northeast-1

# Verify
aws s3 ls
```

### 5.2 Configure CORS (for browser access)

Create `cors.json`:
```json
{
    "CORSRules": [
        {
            "AllowedHeaders": ["*"],
            "AllowedMethods": ["GET", "PUT", "POST", "DELETE"],
            "AllowedOrigins": ["http://localhost:3000", "https://your-domain.com"],
            "ExposeHeaders": ["ETag"]
        }
    ]
}
```

Apply CORS:
```bash
aws s3api put-bucket-cors \
    --bucket character-audio-YOUR_ACCOUNT_ID \
    --cors-configuration file://cors.json
```

---

## Step 6: Test Amazon Polly (TTS)

```bash
# Synthesize Japanese speech
aws polly synthesize-speech \
    --output-format mp3 \
    --voice-id Takumi \
    --engine neural \
    --language-code ja-JP \
    --text "こんにちは、私はAIアシスタントです。" \
    --region ap-northeast-1 \
    hello.mp3

# Play the audio (macOS)
afplay hello.mp3

# Available Japanese voices
aws polly describe-voices --language-code ja-JP --region ap-northeast-1 \
    --query "Voices[?SupportedEngines[?contains(@, 'neural')]].{Id:Id,Gender:Gender}" \
    --output table
```

Output:
```
---------------------------
|     DescribeVoices      |
+--------+----------------+
| Gender |      Id        |
+--------+----------------+
|  Male  |  Takumi        |
|  Female|  Mizuki        |
|  Female|  Kazuha        |
+--------+----------------+
```

---

## Step 7: Test Amazon Transcribe (STT)

### 7.1 Upload Test Audio to S3

```bash
# Upload audio file
aws s3 cp test-audio.wav s3://character-audio-YOUR_ACCOUNT_ID/test/
```

### 7.2 Start Transcription Job

```bash
# Start transcription
aws transcribe start-transcription-job \
    --transcription-job-name test-job-001 \
    --language-code ja-JP \
    --media MediaFileUri=s3://character-audio-YOUR_ACCOUNT_ID/test/test-audio.wav \
    --region ap-northeast-1

# Check status
aws transcribe get-transcription-job \
    --transcription-job-name test-job-001 \
    --region ap-northeast-1 \
    --query "TranscriptionJob.TranscriptionJobStatus"
```

---

## Step 8: Start Local PostgreSQL

```bash
cd /Volumes/RD/character

# Start PostgreSQL with pgvector
docker-compose up -d postgres

# Verify it's running
docker-compose ps

# Run migrations
cd apps/api
source .venv/bin/activate
python -m db.migrations.run
```

---

## Step 9: Create Environment File

Create `apps/api/.env`:

```bash
# Database (Local)
DATABASE_URL=postgresql://character:character@localhost:5432/character

# AWS Credentials
AWS_REGION=ap-northeast-1
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# S3 Bucket
S3_BUCKET_AUDIO=character-audio-YOUR_ACCOUNT_ID

# Bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Polly (Japanese TTS)
POLLY_VOICE_ID=Takumi
POLLY_ENGINE=neural

# Transcribe (Japanese STT)
TRANSCRIBE_LANGUAGE_CODE=ja-JP

# Character
DEFAULT_CHARACTER_NAME=アイ
DEFAULT_CHARACTER_PERSONALITY=親切で知識豊富なAIアシスタント

# CORS
CORS_ORIGINS=["http://localhost:3000"]
DEBUG=true
```

---

## Step 10: Test Full Setup

### Python Test Script

Create `apps/api/test_aws_services.py`:

```python
#!/usr/bin/env python3
"""Test all AWS services used in the project."""

import boto3
import json
from config.settings import get_settings

settings = get_settings()

def test_bedrock():
    """Test Bedrock Claude access."""
    print("Testing Bedrock (Claude)...")
    client = boto3.client('bedrock-runtime', region_name=settings.aws_region)
    
    response = client.invoke_model(
        modelId=settings.bedrock_model_id,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "こんにちは"}]
        })
    )
    
    result = json.loads(response['body'].read())
    print(f"  ✅ Claude response: {result['content'][0]['text'][:50]}...")

def test_polly():
    """Test Polly TTS."""
    print("Testing Polly (TTS)...")
    client = boto3.client('polly', region_name=settings.aws_region)
    
    response = client.synthesize_speech(
        Text="テスト",
        VoiceId=settings.polly_voice_id,
        Engine=settings.polly_engine,
        OutputFormat="mp3",
        LanguageCode="ja-JP"
    )
    
    audio_size = len(response['AudioStream'].read())
    print(f"  ✅ Generated {audio_size} bytes of audio")

def test_s3():
    """Test S3 access."""
    print("Testing S3...")
    client = boto3.client('s3', region_name=settings.aws_region)
    
    # Test write
    client.put_object(
        Bucket=settings.s3_bucket_audio,
        Key="test/hello.txt",
        Body=b"Hello from test!"
    )
    print(f"  ✅ Wrote to s3://{settings.s3_bucket_audio}/test/hello.txt")
    
    # Test read
    response = client.get_object(
        Bucket=settings.s3_bucket_audio,
        Key="test/hello.txt"
    )
    content = response['Body'].read().decode()
    print(f"  ✅ Read: {content}")
    
    # Cleanup
    client.delete_object(Bucket=settings.s3_bucket_audio, Key="test/hello.txt")

def test_transcribe():
    """Test Transcribe access (just check API)."""
    print("Testing Transcribe...")
    client = boto3.client('transcribe', region_name=settings.aws_region)
    
    # Just list jobs to verify access
    response = client.list_transcription_jobs(MaxResults=1)
    print(f"  ✅ Transcribe API accessible")

def main():
    print("=" * 50)
    print("AWS Services Test")
    print("=" * 50)
    print(f"Region: {settings.aws_region}")
    print(f"Model: {settings.bedrock_model_id}")
    print(f"S3 Bucket: {settings.s3_bucket_audio}")
    print("=" * 50)
    
    try:
        test_bedrock()
        test_polly()
        test_s3()
        test_transcribe()
        print("=" * 50)
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
```

Run the test:
```bash
cd apps/api
source .venv/bin/activate
python test_aws_services.py
```

---

# Part 2: EC2 Production Setup

## Architecture on EC2

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐      ┌─────────────┐                     │
│   │ CloudFront  │      │     S3      │                     │
│   │    (CDN)    │      │   (Audio)   │                     │
│   └──────┬──────┘      └─────────────┘                     │
│          │                    ▲                             │
│          ▼                    │                             │
│   ┌─────────────────────────────────────────────┐          │
│   │                 EC2 Instance                │          │
│   │  ┌───────────────────────────────────────┐  │          │
│   │  │              Docker                    │  │          │
│   │  │  ┌─────────┐  ┌─────────────────────┐ │  │          │
│   │  │  │  Nginx  │  │    FastAPI App      │ │  │          │
│   │  │  │  :80    │──│    + CrewAI         │ │  │          │
│   │  │  └─────────┘  └─────────────────────┘ │  │          │
│   │  └───────────────────────────────────────┘  │          │
│   └──────────────────────────────┬──────────────┘          │
│                                  │                          │
│   ┌──────────────┐   ┌───────────▼───────────┐             │
│   │   Bedrock    │   │    RDS PostgreSQL     │             │
│   │   Polly      │   │    (with pgvector)    │             │
│   │   Transcribe │   └───────────────────────┘             │
│   └──────────────┘                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Create VPC and Security Groups

### 1.1 Create VPC (or use default)

```bash
# Create VPC
aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=character-vpc}]' \
    --region ap-northeast-1
```

### 1.2 Create Security Group for EC2

```bash
# Create security group
aws ec2 create-security-group \
    --group-name character-ec2-sg \
    --description "Security group for Character AI EC2" \
    --vpc-id vpc-XXXXX \
    --region ap-northeast-1

# Allow SSH (restrict IP in production!)
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXX \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

# Allow HTTP
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXX \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

# Allow HTTPS
aws ec2 authorize-security-group-ingress \
    --group-id sg-XXXXX \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0
```

---

## Step 2: Create IAM Role for EC2

### 2.1 Create Trust Policy

Create `ec2-trust-policy.json`:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

### 2.2 Create Role

```bash
# Create role
aws iam create-role \
    --role-name character-ec2-role \
    --assume-role-policy-document file://ec2-trust-policy.json

# Attach policies
aws iam attach-role-policy \
    --role-name character-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess

aws iam attach-role-policy \
    --role-name character-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

aws iam attach-role-policy \
    --role-name character-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonPollyFullAccess

aws iam attach-role-policy \
    --role-name character-ec2-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonTranscribeFullAccess

# Create instance profile
aws iam create-instance-profile \
    --instance-profile-name character-ec2-profile

aws iam add-role-to-instance-profile \
    --instance-profile-name character-ec2-profile \
    --role-name character-ec2-role
```

---

## Step 3: Create RDS PostgreSQL

### 3.1 Create DB Security Group

```bash
# Create security group for RDS
aws ec2 create-security-group \
    --group-name character-rds-sg \
    --description "Security group for Character AI RDS" \
    --vpc-id vpc-XXXXX

# Allow PostgreSQL from EC2 security group only
aws ec2 authorize-security-group-ingress \
    --group-id sg-RDS-XXXXX \
    --protocol tcp \
    --port 5432 \
    --source-group sg-EC2-XXXXX
```

### 3.2 Create RDS Instance

```bash
aws rds create-db-instance \
    --db-instance-identifier character-postgres \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 16.1 \
    --master-username character \
    --master-user-password YOUR_SECURE_PASSWORD \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-RDS-XXXXX \
    --db-name character \
    --region ap-northeast-1 \
    --no-publicly-accessible
```

### 3.3 Enable pgvector Extension

After RDS is created, connect and run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Step 4: Launch EC2 Instance

### 4.1 Create Key Pair

```bash
aws ec2 create-key-pair \
    --key-name character-key \
    --query 'KeyMaterial' \
    --output text > character-key.pem

chmod 400 character-key.pem
```

### 4.2 Launch Instance

```bash
aws ec2 run-instances \
    --image-id ami-0d52744d6551d851e \
    --instance-type t3.medium \
    --key-name character-key \
    --security-group-ids sg-EC2-XXXXX \
    --iam-instance-profile Name=character-ec2-profile \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=character-api}]' \
    --region ap-northeast-1
```

### 4.3 Allocate Elastic IP

```bash
# Allocate EIP
aws ec2 allocate-address --domain vpc

# Associate with instance
aws ec2 associate-address \
    --instance-id i-XXXXX \
    --allocation-id eipalloc-XXXXX
```

---

## Step 5: Setup EC2 Instance

### 5.1 Connect to EC2

```bash
ssh -i character-key.pem ec2-user@YOUR_ELASTIC_IP
```

### 5.2 Install Docker

```bash
# Update system
sudo dnf update -y

# Install Docker
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again for group change
exit
```

### 5.3 Clone and Configure Project

```bash
ssh -i character-key.pem ec2-user@YOUR_ELASTIC_IP

# Create app directory
sudo mkdir -p /opt/character
sudo chown ec2-user:ec2-user /opt/character
cd /opt/character

# Clone repository (or upload your code)
git clone https://github.com/YOUR_REPO/character.git .

# Create environment file
cat > .env << 'EOF'
# Database (RDS)
DATABASE_URL=postgresql://character:YOUR_PASSWORD@character-postgres.XXXXX.ap-northeast-1.rds.amazonaws.com:5432/character

# AWS (using IAM role - no keys needed!)
AWS_REGION=ap-northeast-1

# S3
S3_BUCKET_AUDIO=character-audio-YOUR_ACCOUNT_ID

# Bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Polly
POLLY_VOICE_ID=Takumi
POLLY_ENGINE=neural

# Transcribe
TRANSCRIBE_LANGUAGE_CODE=ja-JP

# Application
DEBUG=false
CORS_ORIGINS=["https://your-domain.com"]
EOF
```

### 5.4 Deploy with Docker Compose

```bash
# Build and start
docker-compose -f docker-compose.prod.yml up -d --build

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f api
```

---

## Step 6: Configure Nginx (SSL)

### 6.1 Install Certbot

```bash
sudo dnf install -y certbot python3-certbot-nginx
```

### 6.2 Get SSL Certificate

```bash
sudo certbot --nginx -d your-domain.com
```

### 6.3 Update Nginx Config

```nginx
# /etc/nginx/conf.d/character.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /socket.io/ {
        proxy_pass http://localhost:8000/socket.io/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Quick Reference

### Local Development Commands

```bash
# Start all services
cd /Volumes/RD/character
docker-compose up -d postgres
cd apps/api && source .venv/bin/activate
uvicorn main:app --reload

# In another terminal
cd apps/web && npm run dev
```

### EC2 Commands

```bash
# SSH to EC2
ssh -i character-key.pem ec2-user@YOUR_IP

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Restart services
docker-compose -f docker-compose.prod.yml restart

# Update deployment
git pull
docker-compose -f docker-compose.prod.yml up -d --build
```

### AWS Service Endpoints

| Service | Endpoint |
|---------|----------|
| Bedrock Runtime | `bedrock-runtime.ap-northeast-1.amazonaws.com` |
| Polly | `polly.ap-northeast-1.amazonaws.com` |
| Transcribe | `transcribe.ap-northeast-1.amazonaws.com` |
| S3 | `s3.ap-northeast-1.amazonaws.com` |

---

## Estimated Monthly Costs

| Service | Local Dev | EC2 Production |
|---------|-----------|----------------|
| EC2 (t3.medium) | - | ~$30 |
| RDS (db.t3.micro) | - | ~$15 |
| Bedrock (Claude) | ~$10-50 | ~$50-100 |
| Polly | ~$5 | ~$16 |
| Transcribe | ~$5 | ~$15 |
| S3 | ~$1 | ~$5 |
| **Total** | **~$20-60** | **~$130-180** |

---

## Troubleshooting

### "Unable to locate credentials"
- **Local**: Run `aws configure` and enter your keys
- **EC2**: Ensure IAM role is attached to the instance

### "Access Denied" for Bedrock
- Complete Anthropic usage form in Bedrock console
- Check IAM policy includes `bedrock:InvokeModel`

### RDS Connection Timeout
- Check security group allows traffic from EC2
- Verify RDS is in same VPC as EC2

### Polly/Transcribe Errors
- Verify region is `ap-northeast-1`
- Check language code is `ja-JP`

