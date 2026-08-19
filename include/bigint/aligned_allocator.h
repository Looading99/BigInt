#include <limits>
#include <new>

namespace bigint {

template<typename T, std::size_t Align = 32> struct AlignedAllocator {
    static_assert(Align >= alignof(T), "Alignment must be at least alignof(T)");
    static_assert((Align & (Align - 1)) == 0, "Alignment must be a power of two");

    using value_type = T;

    AlignedAllocator() = default;

    template<typename U> AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    auto allocate(std::size_t n) -> T* {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        void* p = ::operator new(n * sizeof(T), std::align_val_t(Align));
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t(Align)); }

    template<typename U> struct rebind {
        using other = AlignedAllocator<U, Align>;
    };
};

template<typename T, typename U, std::size_t Align>
auto operator==(const AlignedAllocator<T, Align>&, const AlignedAllocator<U, Align>&) noexcept
    -> bool {
    return true;
}

template<typename T, typename U, std::size_t Align>
auto operator!=(const AlignedAllocator<T, Align>&, const AlignedAllocator<U, Align>&) noexcept
    -> bool {
    return false;
}

}  // namespace bigint
