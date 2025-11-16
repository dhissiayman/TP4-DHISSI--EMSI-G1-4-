package ma.emsi.dhissiayman.tp4.tp4dhissiayman.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import jakarta.enterprise.context.Dependent;
import ma.emsi.dhissiayman.tp4.tp4dhissiayman.assistant.Assistant;

import java.io.Serializable;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

@Dependent
public class LlmClient implements Serializable {

    private static final long serialVersionUID = 1L;

    // --------- ÉTAT SÉRIALISABLE ---------
    private String systemRole;

    // --------- COMPOSANTS NON SÉRIALISABLES ---------
    private transient ChatMemory chatMemory;
    private transient Assistant assistant;
    private transient ChatModel model;
    private transient String apiKey;

    // --------- COMPOSANTS RAG PARTAGÉS (UNE SEULE INGESTION POUR TOUTE L'APPLI) ---------
    private static transient RetrievalAugmentor retrievalAugmentor;
    private static transient boolean ragInitialized = false;

    // =========================
    //  RAG : ingestion des PDFs
    // =========================

    /**
     * Ingestion d’un PDF dans un EmbeddingStore (logique similaire à tes tests).
     */
    private static EmbeddingStore<TextSegment> ingestPdfAsEmbeddingStore(
            Class<?> resourceOwner,
            String resourceName,
            EmbeddingModel embeddingModel) {

        try {
            URL resource = resourceOwner.getClassLoader().getResource(resourceName);
            if (resource == null) {
                throw new IllegalStateException("Le fichier " + resourceName + " n'a pas été trouvé dans resources.");
            }
            Path pdfPath = Paths.get(resource.toURI());

            DocumentParser parser = new ApacheTikaDocumentParser();
            Document document = FileSystemDocumentLoader.loadDocument(pdfPath, parser);

            DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);
            List<TextSegment> segments = splitter.split(document);

            Response<List<Embedding>> response = embeddingModel.embedAll(segments);
            List<Embedding> embeddings = response.content();

            EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
            store.addAll(embeddings, segments);

            System.out.println("Ingestion terminée pour " + resourceName + " : "
                    + segments.size() + " segments enregistrés.");

            return store;
        } catch (URISyntaxException e) {
            throw new RuntimeException("Erreur lors du chargement du PDF " + resourceName, e);
        }
    }

    /**
     * Initialisation statique du RAG multi-PDF + Web (Tavily).
     * Appelée une seule fois (synchronisée) pour toute la JVM.
     */
    private static synchronized void ensureRagInitialized(String apiKeyGemini) {
        if (ragInitialized && retrievalAugmentor != null) {
            return;
        }

        // Modèle d'embedding partagé
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // --------- PHASE 1 : ingestion de 2 PDFs ---------
        EmbeddingStore<TextSegment> iaStore =
                ingestPdfAsEmbeddingStore(LlmClient.class, "langchain4j.pdf", embeddingModel);

        EmbeddingStore<TextSegment> autreStore =
                ingestPdfAsEmbeddingStore(LlmClient.class, "QCM_MAD-AI_COMPLET.pdf", embeddingModel);

        // --------- PHASE 2 : ContentRetrievers sur les PDFs ---------
        ContentRetriever iaRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(iaStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        ContentRetriever autreRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(autreStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // --------- PHASE 3 : ContentRetriever Web (Tavily) ---------
        String tavilyKey = System.getenv("TAVILY_API_KEY");
        if (tavilyKey == null || tavilyKey.isBlank()) {
            throw new IllegalStateException("La variable d'environnement TAVILY_API_KEY n'est pas définie");
        }

        WebSearchEngine tavilyEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever webRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(tavilyEngine)
                        // tu peux ajouter .maxResults(), .includeRawContent(), etc. si besoin
                        .build();

        // --------- PHASE 4 : QueryRouter combinant PDF1 + PDF2 + Web ---------
        // DefaultQueryRouter interroge tous les ContentRetrievers fournis.
        QueryRouter queryRouter = new DefaultQueryRouter(
                iaRetriever,
                autreRetriever,
                webRetriever
        );

        // --------- PHASE 5 : RetrievalAugmentor final ---------
        retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        ragInitialized = true;
    }

    // =========================
    //  INIT PARESSEUSE DU BEAN
    // =========================
    private void ensureInit() {
        if (assistant != null) {
            return;
        }

        this.apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("GEMINI_KEY manquante.");
        }

        // Modèle de chat pour l'assistant Web
        this.model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // Initialisation RAG (2 PDFs + Tavily Web)
        ensureRagInitialized(this.apiKey);

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // Remettre le rôle système s'il existait
        if (this.systemRole != null && !this.systemRole.isBlank()) {
            this.chatMemory.add(SystemMessage.from(this.systemRole));
        }

        // Assistant LangChain4j avec :
        //  - chatModel Gemini
        //  - mémoire
        //  - RetrievalAugmentor (RAG multi-PDF + Web Tavily)
        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(this.model)
                .chatMemory(this.chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();
    }

    // =========================
    //  API UTILISÉE PAR LA WEBAPP
    // =========================

    public void setSystemRole(String role) {
        this.systemRole = role;
        ensureInit();
        this.chatMemory.clear();
        if (role != null && !role.isBlank()) {
            this.chatMemory.add(SystemMessage.from(role));
        }
    }

    public String ask(String prompt) {
        ensureInit();
        return assistant.chat(prompt);
    }
}
