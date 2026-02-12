FROM node:20-alpine

WORKDIR /app/frontend

COPY frontend/package.json ./package.json
RUN npm install

COPY frontend /app/frontend

EXPOSE 5173

CMD ["npm", "run", "dev"]
