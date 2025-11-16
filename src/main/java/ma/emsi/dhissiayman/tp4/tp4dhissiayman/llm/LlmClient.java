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
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.enterprise.context.Dependent;
import ma.emsi.dhissiayman.tp4.tp4dhissiayman.assistant.Assistant;

import java.io.Serializable;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

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
     * Ingestion d’un PDF dans un EmbeddingStore (reprend la logique de Test3Routage).
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
     * Initialisation statique du RAG multi-documents avec routage LLM.
     * Appelée une seule fois (synchronisée) pour toute la JVM.
     */
    private static synchronized void ensureRagInitialized(String apiKey) {
        if (ragInitialized && retrievalAugmentor != null) {
            return;
        }

        // Modèle de chat utilisé pour le routage des requêtes (et éventuellement pour d'autres usages)
        ChatModel routingChatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // Modèle d'embedding partagé par les deux sources
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // --------- PHASE 1 : ingestion des 2 PDFs ---------
        // Adapte les noms aux PDF que tu as mis dans src/main/resources
        EmbeddingStore<TextSegment> iaStore =
                ingestPdfAsEmbeddingStore(LlmClient.class, "langchain4j.pdf", embeddingModel);

        EmbeddingStore<TextSegment> autreStore =
                ingestPdfAsEmbeddingStore(LlmClient.class, "QCM_MAD-AI_COMPLET.pdf", embeddingModel);

        // --------- PHASE 2 : 2 ContentRetrievers ---------
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

        // Map des descriptions en langage naturel pour le routage
        Map<ContentRetriever, String> retrieverToDescription = Map.of(
                iaRetriever,
                "Documents de cours sur l'IA, les LLM, le RAG, LangChain4j, etc.",
                autreRetriever,
                "Documents qui ne parlent pas directement d'IA (autres matières / autres sujets)."
        );

        // QueryRouter basé sur le LLM (exactement comme dans Test3Routage)
        QueryRouter queryRouter = LanguageModelQueryRouter.builder()
                .chatModel(routingChatModel)
                .retrieverToDescription(retrieverToDescription)
                .build();

        // RetrievalAugmentor final basé sur ce QueryRouter
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

        // Modèle de chat pour les réponses de l'assistant (côté Web)
        this.model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // Initialisation RAG (ingestion PDF + routage) une seule fois
        ensureRagInitialized(this.apiKey);

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // Si un rôle système existait avant passivation, on le remet
        if (this.systemRole != null && !this.systemRole.isBlank()) {
            this.chatMemory.add(SystemMessage.from(this.systemRole));
        }

        // Assistant LangChain4j avec :
        //  - chatModel Gemini
        //  - mémoire
        //  - RetrievalAugmentor (RAG + routage multi-PDF)
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
        // assistant peut être null après passivation -> réinit
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
