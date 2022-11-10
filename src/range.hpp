#pragma once

template <typename Value>
struct BlockedRange {
	using size_type = size_t;
	using const_iterator = Value;

	BlockedRange(Value begin, Value end): _begin(begin), _end(end) { }

	size_type size() const { return end() - begin(); }
	bool empty() const { return !(begin() < end()); }

	const_iterator begin() const { return _begin; }
	const_iterator end() const { return _end; }
private:
	Value _begin;
	Value _end;
};

template <typename Value, size_t D>
struct BlockedRangeND {
	using size_type = size_t;
	static constexpr size_t dimension = D;

	// using const_iterator = std::array<Value, D>;
	struct const_iterator {
		using size_type = size_t;
		using value_type = std::array<Value, D>;

		BlockedRangeND range;
		value_type value;

		const_iterator & operator++() {
			for (size_t d = D-1 ; d < D ; --d) { // stop by overflow
				if (value[d] < range.upper(d)) {
					++value[d];
					return *this;
				} else {
					value[d] = 0;
				}
			}
		}
		// const_iterator operator++(int);
		value_type operator*() const { return value; }
	};

	BlockedRangeND(std::array<Value, D> const & lower, std::array<Value, D> const & upper): _lower(lower), _upper(upper) { }

	// begin();
	// end();

private:
	std::array<Value, D> _lower;
	std::array<Value, D> _upper;

};
