# Stage 1: Builder
FROM node:18 AS builder

WORKDIR /app

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for Python
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Generate Prisma Client
RUN npx prisma generate

# Build the application
RUN npm run build

# Stage 2: Production
FROM node:18

WORKDIR /app

ENV NODE_ENV=production
# Ensure Python is available in production for child_process
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install only production dependencies
COPY package*.json ./
RUN npm install --only=production

# Copy built assets and prisma
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/prisma ./prisma
COPY --from=builder /app/docs ./docs
# Copy Python scripts and models (ensure src is available if script is in src)
# The controller spawns 'src/controllers/skinanalysisController.py', so we need it.
# We'll copy the whole src folder to be safe or just the needed scripts. 
# Since 'dist' usually compiles TS->JS, but we have JS files and Python files. 
# If 'npm run build' is just babel, it puts JS in dist. 
# We need to make sure the python script is where the controller looks for it.
# The controller looks in: path.join(__dirname, 'skinanalysisController.py')
# In prod, __dirname will be 'dist/controllers'. 
# So we must COPY the python script to 'dist/controllers' or copy 'src' to 'src'.
# Let's verify where the build puts things. Assuming build is `babel src --out-dir dist`.
# Babel usually ignores non-JS files unless configured. 
# SAFEST BET: Copy the python script manually to the expected location in dist.

COPY --from=builder /app/src/controllers/skinanalysisController.py ./dist/controllers/

# Generate Prisma Client for production
RUN npx prisma generate

# Expose port
EXPOSE 3000

# Start command
CMD ["node", "dist/server.js"]
